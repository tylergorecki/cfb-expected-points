"""
College Football Expected Points Model

A Monte Carlo simulation-based model for calculating expected points in college football
from any given field position. The model uses historical play-by-play data from the 2019
college football season to fit statistical distributions for different play types.

Author: Tyler Gorecki
Data Source: 2019 PFF College Football Play-by-Play Data
"""

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Configuration
# ============================================================================

FIELD_ZONES = {
    'own': (0, 40),
    'midfield': (40, 65),
    'fg_range': (65, 80),
    'red_zone': (80, 100)
}

FUMBLE_OFFENSE_RECOVERY_RATE = 0.5  # Approximate share of fumbles recovered by offense

SCORING = {
    'touchdown': 7,
    'safety': 2,
    'field_goal': 3
}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class PlayTypeDistribution:
    """Statistical distribution parameters for a play type in a field zone."""
    prob: float  # Probability of this play type occurring
    mean: float
    std: float
    
    # For gamma distribution (deep pass)
    alpha: Optional[float] = None
    beta: Optional[float] = None


@dataclass
class TurnoverRates:
    """Turnover probabilities for a play type in a field zone (may be bucketed)."""
    interception_rate: float  # Probability of interception (passes only)
    fumble_rate: float  # Probability of fumble event (lost or not)


@dataclass
class FieldZoneParams:
    """Play type parameters for a specific field zone."""
    runs: PlayTypeDistribution
    short_pass: PlayTypeDistribution
    deep_pass: PlayTypeDistribution
    sack: PlayTypeDistribution
    run_turnovers: Dict[str, TurnoverRates]
    short_pass_turnovers: Dict[str, TurnoverRates]
    deep_pass_turnovers: Dict[str, TurnoverRates]
    punt_distance: float


# ============================================================================
# Data Preparation
# ============================================================================

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load and prepare play-by-play data.
    
    Args:
        filepath: Path to the PFF all plays CSV file
        
    Returns:
        Prepared DataFrame with field position and play type classifications
    """
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} plays from {filepath}")

    # Coerce kick depth to numeric to avoid NaN means for punts
    if 'pff_KICKDEPTH' in df.columns:
        df['pff_KICKDEPTH'] = pd.to_numeric(df['pff_KICKDEPTH'], errors='coerce')
    
    # Standardize field position encoding
    df['field_pos'] = df['pff_FIELDPOSITION'].apply(lambda x: 
        abs(x) if x < 0 else 100 - x
    )
    
    # Classify field zones
    df['field_zone'] = pd.cut(df['field_pos'], 
                               bins=[0, 40, 65, 80, 100],
                               labels=['own', 'midfield', 'fg_range', 'red_zone'],
                               include_lowest=True)
    
    # Classify play types
    df['play_type'] = df.apply(_classify_play_type, axis=1)
    
    # Detect turnovers (dataset uses string codes when events occur)
    df['is_interception'] = df['pff_INTERCEPTION'].notna()
    df['is_fumble_event'] = df['pff_FUMBLE'].notna()
    df['fumble_recovered_by_offense'] = df['pff_FUMBLERECOVERY'].notna()
    df['is_fumble_turnover'] = df['is_fumble_event'] & (~df['fumble_recovered_by_offense'])
    df['is_turnover'] = df['is_interception'] | df['is_fumble_turnover']
    
    return df


def _classify_play_type(row: pd.Series) -> str:
    """Classify a play as run, sack, short pass, deep pass, or other."""
    # Special teams: detect field goals/punts/kickoffs before run/pass logic
    st_type = str(row.get('pff_SPECIALTEAMSTYPE', '') or '').upper()
    if st_type == 'FIELD GOAL':
        return 'field_goal'
    if st_type == 'PUNT':
        return 'special_teams'
    if st_type == 'KICKOFF':
        return 'kickoff'
    if st_type == 'EXTRA POINT':
        return 'extra_point'

    if pd.notna(row.get('pff_PENALTY')) and row['pff_PENALTY'] != '':
        return 'penalty'
    if pd.notna(row.get('pff_SACK')) and row['pff_SACK'] != '':
        return 'sack'
    if pd.notna(row.get('pff_RUNPASS')):
        if row['pff_RUNPASS'] == 'R':
            return 'run'
        elif row['pff_RUNPASS'] == 'P':
            is_deep = row.get('pff_DEEPPASS', 0) == 1
            is_complete = pd.isna(row.get('pff_INCOMPLETIONTYPE')) or row['pff_INCOMPLETIONTYPE'] == ''
            
            if is_deep:
                return 'deep_pass' if is_complete else 'deep_pass_incomplete'
            else:
                return 'short_pass' if is_complete else 'short_pass_incomplete'
        elif row['pff_RUNPASS'] == 'X':
            return 'special_teams'
    return 'other'


# ============================================================================
# Field Goal Modeling
# ============================================================================

def fit_field_goal_model(fg_data: pd.DataFrame) -> Tuple[float, float]:
    """
    Fit logistic regression to model field goal success probability.
    
    Args:
        fg_data: DataFrame with field goal attempts (must have 'field_pos' and 'fg_result')
        
    Returns:
        Tuple of (intercept, slope) coefficients for logistic model
    """
    from sklearn.linear_model import LogisticRegression
    
    X = fg_data['field_pos'].values.reshape(-1, 1)
    y = fg_data['fg_result'].values
    
    model = LogisticRegression()
    model.fit(X, y)
    
    logger.info(f"Field goal model fitted with intercept={model.intercept_[0]:.4f}, "
                f"slope={model.coef_[0][0]:.4f}")
    
    return model.intercept_[0], model.coef_[0][0]


def field_goal_success_prob(field_pos: float, intercept: float, slope: float) -> float:
    """
    Calculate field goal success probability from field position using logistic model.
    
    Args:
        field_pos: Field position (0-100)
        intercept: Logistic model intercept
        slope: Logistic model slope
        
    Returns:
        Success probability (0-1)
    """
    logit = intercept + slope * field_pos
    return 1 / (1 + np.exp(-logit))


# ============================================================================
# Play Outcome Simulation
# ============================================================================

def _down_bucket(down: int) -> str:
    return str(int(np.clip(down, 1, 4)))


def _ytg_bucket(yards_to_go: int) -> str:
    if yards_to_go <= 2:
        return 'short'
    if yards_to_go <= 5:
        return 'medium'
    if yards_to_go <= 10:
        return 'long'
    return 'very_long'


def _lookup_turnover_rates(play_type: str, down: int, yards_to_go: int, params: FieldZoneParams) -> TurnoverRates:
    key = f"{_down_bucket(down)}:{_ytg_bucket(yards_to_go)}"
    table = {
        'run': params.run_turnovers,
        'short_pass': params.short_pass_turnovers,
        'deep_pass': params.deep_pass_turnovers,
        'sack': {'default': TurnoverRates(0.0, 0.0)}
    }.get(play_type, {})
    return table.get(key, table.get('default', TurnoverRates(0.0, 0.0)))


def sample_yards_gained(field_pos: float, down: int, yards_to_go: int, zone_params: dict) -> Tuple[int, bool]:
    """
    Sample yards gained from appropriate distribution based on play type probabilities.
    
    Uses hierarchical sampling: first determines play type based on field zone probabilities,
    then samples yards from the corresponding distribution. Also samples for turnovers.
    
    Args:
        field_pos: Field position (0-100)
        down: Current down (1-4)
        zone_params: Dictionary mapping field zones to FieldZoneParams
        
    Returns:
        Tuple of (yards_gained, is_turnover)
    """
    zone = _get_field_zone(field_pos)
    params = zone_params[zone]
    
    # Get probabilities
    probs = [
        params.runs.prob,
        params.short_pass.prob,
        params.deep_pass.prob,
        params.sack.prob
    ]
    probs = np.array(probs) / np.sum(probs)  # Normalize
    
    play_type = np.random.choice(['run', 'short_pass', 'deep_pass', 'sack'], 
                                 p=probs)
    to_rates = _lookup_turnover_rates(play_type, down, yards_to_go, params)
    
    # Sample yards
    if play_type == 'run':
        yards = _sample_from_exponential_mixture(params.runs, positive=True)
    elif play_type == 'short_pass':
        sd = params.short_pass.std if params.short_pass.std and params.short_pass.std > 0 else 1.0
        yards = int(np.random.normal(params.short_pass.mean, sd))
    elif play_type == 'deep_pass':
        if params.deep_pass.alpha and params.deep_pass.beta:
            yards = int(np.random.gamma(params.deep_pass.alpha, 1/params.deep_pass.beta))
        else:
            sd = params.deep_pass.std if params.deep_pass.std and params.deep_pass.std > 0 else 5.0
            yards = int(np.random.normal(params.deep_pass.mean, sd))
    else:  # sack
        sack_scale = abs(params.sack.mean) if params.sack.mean != 0 else 1.0
        yards = -int(np.random.exponential(sack_scale))
        to_rates = TurnoverRates(interception_rate=0.0, fumble_rate=params.sack.prob * 0.05)  # Minimal fumble on sack
    
    # Determine if turnover
    is_turnover = False
    if play_type in ['short_pass', 'deep_pass']:
        is_turnover = np.random.random() < to_rates.interception_rate
    elif play_type == 'run':
        fumble_event = np.random.random() < to_rates.fumble_rate
        if fumble_event:
            offense_recovers = np.random.random() < FUMBLE_OFFENSE_RECOVERY_RATE
            is_turnover = not offense_recovers
    elif play_type == 'sack':
        fumble_event = np.random.random() < to_rates.fumble_rate
        if fumble_event:
            offense_recovers = np.random.random() < FUMBLE_OFFENSE_RECOVERY_RATE
            is_turnover = not offense_recovers
    
    return yards, is_turnover


def _sample_from_exponential_mixture(dist: PlayTypeDistribution, positive: bool = True) -> int:
    """Sample from mixture of exponential distributions for runs (positive and negative outcomes)."""
    mean = dist.mean if positive else abs(dist.mean)
    alpha = 1 / mean if mean != 0 else 0.1
    sample = int(np.random.exponential(1/alpha))
    return sample if positive else -sample


def _get_field_zone(field_pos: float) -> str:
    """Get field zone name from field position."""
    if field_pos < 40:
        return 'own'
    elif field_pos < 65:
        return 'midfield'
    elif field_pos < 80:
        return 'fg_range'
    else:
        return 'red_zone'


# ============================================================================
# Expected Points Calculation
# ============================================================================

class ExpectedPointsCalculator:
    """Main class for calculating expected points using Monte Carlo simulation."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the calculator with prepared play-by-play data.
        
        Args:
            data: Prepared DataFrame from load_and_prepare_data()
        """
        self.data = data
        self.zone_params = self._fit_zone_parameters()
        self.fg_intercept, self.fg_slope = self._fit_field_goal_model()
        self.punt_distances = self._fit_punt_distances()
        
    def _fit_zone_parameters(self) -> dict:
        """Fit distribution parameters for each play type in each field zone."""
        zone_params = {}
        
        for zone_name, (zone_min, zone_max) in FIELD_ZONES.items():
            zone_data = self.data[
                (self.data['field_pos'] >= zone_min) & 
                (self.data['field_pos'] < zone_max)
            ]
            
            zone_params[zone_name] = FieldZoneParams(
                runs=self._fit_play_distribution(zone_data, 'run'),
                short_pass=self._fit_play_distribution(zone_data, 'short_pass'),
                deep_pass=self._fit_play_distribution(zone_data, 'deep_pass'),
                sack=self._fit_play_distribution(zone_data, 'sack'),
                run_turnovers=self._fit_turnover_rates(zone_data, 'run'),
                short_pass_turnovers=self._fit_turnover_rates(zone_data, 'short_pass'),
                deep_pass_turnovers=self._fit_turnover_rates(zone_data, 'deep_pass'),
                punt_distance=self._calc_punt_distance(zone_data)
            )
            
            logger.info(f"Fitted parameters for {zone_name} zone")
        
        return zone_params
    
    def _fit_play_distribution(self, zone_data: pd.DataFrame, play_type: str) -> PlayTypeDistribution:
        """Fit distribution parameters for a specific play type."""
        play_data = zone_data[zone_data['play_type'] == play_type]['pff_GAINLOSSNET']
        
        if len(play_data) == 0:
            # Handle case with no data for this play type
            return PlayTypeDistribution(prob=0.01, mean=0, std=1)
        
        prob = len(play_data) / len(zone_data) if len(zone_data) > 0 else 0
        mean = play_data.mean()
        std = play_data.std()
        
        # For deep pass (gamma distribution)
        alpha = None
        beta = None
        if play_type == 'deep_pass':
            if std > 0:
                alpha = mean ** 2 / (std ** 2)
                beta = mean / (std ** 2)
        
        return PlayTypeDistribution(prob=prob, mean=mean, std=std, alpha=alpha, beta=beta)
    
    def _fit_turnover_rates(self, zone_data: pd.DataFrame, play_type: str) -> Dict[str, TurnoverRates]:
        """Fit turnover probabilities bucketed by down and distance for a specific play type in a zone."""
        play_data = zone_data[zone_data['play_type'] == play_type].copy()
        if len(play_data) == 0:
            return {'default': TurnoverRates(interception_rate=0.0, fumble_rate=0.0)}

        # Prepare buckets
        play_data['down_bucket'] = play_data['pff_DOWN'].fillna(1).clip(1, 4).astype(int)
        play_data['ytg_bucket'] = play_data['pff_DISTANCE'].fillna(10).apply(_ytg_bucket)

        buckets: Dict[str, TurnoverRates] = {}

        for (down_bucket, ytg_bucket), grp in play_data.groupby(['down_bucket', 'ytg_bucket']):
            key = f"{int(down_bucket)}:{ytg_bucket}"
            int_rate = 0.0
            if play_type in ['short_pass', 'deep_pass']:
                int_rate = grp['is_interception'].mean()
            fum_rate = grp['is_fumble_event'].mean()
            buckets[key] = TurnoverRates(interception_rate=float(int_rate or 0.0),
                                         fumble_rate=float(fum_rate or 0.0))

        # Global fallback
        overall_int = play_data['is_interception'].mean() if play_type in ['short_pass', 'deep_pass'] else 0.0
        overall_fum = play_data['is_fumble_event'].mean()
        buckets['default'] = TurnoverRates(interception_rate=float(overall_int or 0.0),
                                           fumble_rate=float(overall_fum or 0.0))

        return buckets
    
    def _calc_punt_distance(self, zone_data: pd.DataFrame) -> float:
        """Calculate average punt distance for a field zone."""
        punt_data = zone_data[zone_data['play_type'] == 'special_teams']
        if len(punt_data) == 0:
            return 35.0  # Default average punt distance
        punt_mean = punt_data['pff_KICKDEPTH'].mean()
        return 35.0 if pd.isna(punt_mean) else float(punt_mean)
    
    def _fit_field_goal_model(self) -> Tuple[float, float]:
        """Fit logistic regression for field goal success."""
        fg_data = self.data[
            self.data['play_type'].str.contains('field_goal', na=False)
        ].copy()
        
        if len(fg_data) == 0:
            logger.warning("No field goal data found")
            return 0, 0
        
        fg_data['fg_result'] = (fg_data['pff_KICKRESULT'].str.contains('MADE', na=False)).astype(int)
        return fit_field_goal_model(fg_data)
    
    def _fit_punt_distances(self) -> dict:
        """Get average punt distances by field zone."""
        punt_data = self.data[self.data['play_type'] == 'special_teams']
        distances = {}
        for zone_name, (zone_min, zone_max) in FIELD_ZONES.items():
            zone_punt_data = punt_data[
                (punt_data['field_pos'] >= zone_min) & 
                (punt_data['field_pos'] < zone_max)
            ]
            if len(zone_punt_data) == 0:
                distances[zone_name] = 35.0
            else:
                mean_val = zone_punt_data['pff_KICKDEPTH'].mean()
                distances[zone_name] = 35.0 if pd.isna(mean_val) else float(mean_val)
        return distances
    
    def calculate_expected_points(self, down: int, field_pos: float, 
                                  yards_to_go: int, num_simulations: int = 100) -> float:
        """
        Calculate expected points from a game situation using Monte Carlo simulation.
        
        Args:
            down: Current down (1-4)
            field_pos: Field position (0-100)
            yards_to_go: Yards needed for first down
            num_simulations: Number of simulations to average
            
        Returns:
            Expected points value
        """
        simulations = [
            self._simulate_drive(down, field_pos, yards_to_go, possession=1)
            for _ in range(num_simulations)
        ]
        return np.mean(simulations)
    
    def _simulate_drive(self, down: int, field_pos: float, 
                       yards_to_go: int, possession: int) -> float:
        """
        Recursively simulate a drive from current game state until scoring.
        
        Args:
            down: Current down (1-4)
            field_pos: Field position (0-100)
            yards_to_go: Yards needed for first down
            possession: Possession indicator (1 for own possession, 0 for opponent)
            
        Returns:
            Points scored by possession team (negative if opponent scores)
        """
        # Base cases: scoring events
        if field_pos >= 100:
            return SCORING['touchdown'] if possession == 1 else -SCORING['touchdown']
        if field_pos <= 0:
            return -SCORING['safety'] if possession == 1 else SCORING['safety']
        
        # Non-fourth down logic
        if down < 4:
            gained, is_turnover = sample_yards_gained(field_pos, down, yards_to_go, self.zone_params)
            field_pos += gained
            
            # Check for turnover before field position checks
            if is_turnover:
                new_possession = 0 if possession == 1 else 1
                new_field_pos = 100 - field_pos
                new_field_pos = max(0, min(100, new_field_pos))
                new_yards = 100 - new_field_pos if 100 - new_field_pos < 10 else 10
                return self._simulate_drive(1, new_field_pos, new_yards, new_possession)
            
            if field_pos >= 100:
                return SCORING['touchdown'] if possession == 1 else -SCORING['touchdown']
            if field_pos <= 0:
                return -SCORING['safety'] if possession == 1 else SCORING['safety']
            
            first_down = gained >= yards_to_go
            if first_down:
                new_down = 1
                new_yards = 100 - field_pos if 100 - field_pos < 10 else 10
            else:
                new_down = down + 1
                new_yards = yards_to_go - gained
            
            return self._simulate_drive(new_down, field_pos, new_yards, possession)
        
        # Fourth down decision logic
        else:
            if yards_to_go > 2:
                # Can attempt field goal or punt
                if field_pos >= 65:
                    return self._attempt_field_goal(field_pos, possession)
                else:
                    return self._punt(field_pos, possession)
            else:
                # Close to first down - can go for it or punt
                if field_pos <= 40:
                    return self._punt(field_pos, possession)
                elif field_pos >= 100:
                    return SCORING['touchdown'] if possession == 1 else -SCORING['touchdown']
                else:
                    # Go for it
                    gained, is_turnover = sample_yards_gained(field_pos, down, yards_to_go, self.zone_params)
                    field_pos += gained
                    
                    if is_turnover:
                        new_possession = 0 if possession == 1 else 1
                        new_field_pos = 100 - field_pos
                        new_field_pos = max(0, min(100, new_field_pos))
                        new_yards = 100 - new_field_pos if 100 - new_field_pos < 10 else 10
                        return self._simulate_drive(1, new_field_pos, new_yards, new_possession)
                    
                    if field_pos >= 100:
                        return SCORING['touchdown'] if possession == 1 else -SCORING['touchdown']
                    
                    first_down = gained >= yards_to_go
                    if first_down:
                        new_down = 1
                        new_yards = 100 - field_pos if 100 - field_pos < 10 else 10
                        return self._simulate_drive(new_down, field_pos, new_yards, possession)
                    else:
                        # Turnover on downs
                        new_possession = 0 if possession == 1 else 1
                        new_field_pos = 100 - field_pos
                        new_yards = 100 - new_field_pos if 100 - new_field_pos < 10 else 10
                        return self._simulate_drive(1, new_field_pos, new_yards, new_possession)
    
    def _attempt_field_goal(self, field_pos: float, possession: int) -> float:
        """Attempt a field goal from current field position."""
        make_prob = field_goal_success_prob(field_pos, self.fg_intercept, self.fg_slope)
        make = np.random.random() < make_prob
        
        if make:
            return SCORING['field_goal'] if possession == 1 else -SCORING['field_goal']
        else:
            # Missed field goal - turnover
            new_possession = 0 if possession == 1 else 1
            new_field_pos = 100 - field_pos
            new_yards = 100 - new_field_pos if 100 - new_field_pos < 10 else 10
            return self._simulate_drive(1, new_field_pos, new_yards, new_possession)
    
    def _punt(self, field_pos: float, possession: int) -> float:
        """Simulate a punt from current field position."""
        zone = _get_field_zone(field_pos)
        punt_dist = self.punt_distances.get(zone, 35.0)
        
        new_field_pos = 100 - (field_pos + punt_dist)
        new_field_pos = max(0, min(100, new_field_pos))
        
        new_possession = 0 if possession == 1 else 1
        new_yards = 100 - new_field_pos if 100 - new_field_pos < 10 else 10
        
        return self._simulate_drive(1, new_field_pos, new_yards, new_possession)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Load and prepare data
    data_path = "2019 PFF All Plays - first1000.csv"
    df = load_and_prepare_data(data_path)
    
    # Initialize calculator
    calc = ExpectedPointsCalculator(df)
    
    # Example calculations
    print("\n=== Expected Points Examples ===")
    ep_1_10_25 = calc.calculate_expected_points(down=1, field_pos=25, yards_to_go=10, num_simulations=100)
    print(f"1st & 10 at own 25: {ep_1_10_25:.2f} points")
    
    ep_1_10_50 = calc.calculate_expected_points(down=1, field_pos=50, yards_to_go=10, num_simulations=100)
    print(f"1st & 10 at midfield: {ep_1_10_50:.2f} points")
    
    ep_1_10_75 = calc.calculate_expected_points(down=1, field_pos=75, yards_to_go=10, num_simulations=100)
    print(f"1st & 10 at opponent 25: {ep_1_10_75:.2f} points")
    
    ep_4_10_64 = calc.calculate_expected_points(down=4, field_pos=64, yards_to_go=10, num_simulations=100)
    print(f"4th & 10 at own 36 (opponent 64): {ep_4_10_64:.2f} points")