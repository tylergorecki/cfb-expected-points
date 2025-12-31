# Expected Points Model - Code Explanation

## Overview

The `epModel.py` module implements a Monte Carlo simulation-based system for calculating expected points in college football from any field position. It processes historical play-by-play data to build statistical models of different play types and uses recursive simulations to estimate expected scoring outcomes.

## Architecture

The codebase is organized into five main sections:

### 1. Data Structures

#### `PlayTypeDistribution`
A dataclass that encapsulates the statistical parameters for a specific play type within a field zone:
- `prob`: Probability of this play type occurring in the zone
- `mean`: Average yards gained
- `std`: Standard deviation of yards gained
- `alpha`, `beta`: Shape and scale parameters for gamma distributions (used for deep passes)

#### `TurnoverRates`
A dataclass containing turnover probabilities for a specific play type:
- `interception_rate`: Probability of interception (passes only)
- `fumble_rate`: Probability of fumble (runs and catches)

#### `FieldZoneParams`
A dataclass containing all play type parameters for a specific field zone:
- `runs`: Distribution parameters for rushing plays
- `short_pass`: Distribution parameters for short passing plays
- `deep_pass`: Distribution parameters for deep passing plays
- `sack`: Distribution parameters for sacks/losses
- `run_turnovers`: Turnover rates for rushing plays
- `short_pass_turnovers`: Turnover rates for short passes
- `deep_pass_turnovers`: Turnover rates for deep passes
- `punt_distance`: Average punt distance from this zone

These structures provide type safety and make the code self-documenting.

### 2. Data Preparation Pipeline

#### `load_and_prepare_data(filepath: str) -> pd.DataFrame`
The entry point for data processing:
1. Loads PFF play-by-play CSV file
2. Standardizes field position encoding (converts to 0-100 scale)
3. Classifies field zones (own side, midfield, field goal range, red zone)
4. Classifies play types using the helper function
5. Detects turnovers (interceptions and fumbles)

**Field Position Standardization**: Converts negative field positions to distances from own end zone (0 = own goal line, 100 = opponent's goal line).

**Turnover Detection**: Identifies interceptions (detected via `pff_INTERCEPTEDPASS` flag) and fumbles (`pff_FUMBLE` flag), marking any play with either as a turnover event.

#### `_classify_play_type(row: pd.Series) -> str`
Implements hierarchical play classification:
1. Checks for penalties
2. Identifies sacks
3. Distinguishes runs from passes
4. Further classifies passes as short or deep
5. Marks special teams plays
6. Handles incomplete passes

The classification hierarchy ensures each play is correctly categorized for distribution fitting.

### 3. Field Goal Modeling

#### `fit_field_goal_model(fg_data: pd.DataFrame) -> Tuple[float, float]`
Uses logistic regression to model field goal success probability:
- **Input**: Field goal attempts with field position and outcome (make/miss)
- **Process**: Fits logistic model with field position as predictor
- **Output**: (intercept, slope) coefficients

The logistic model captures the realistic relationship where field goal success decreases with distance.

#### `field_goal_success_prob(field_pos: float, intercept: float, slope: float) -> float`
Applies the logistic model to compute success probability:
$$P(\text{make}) = \frac{1}{1 + e^{-(\text{intercept} + \text{slope} \times \text{field\_pos})}}$$

### 4. Play Outcome Simulation

#### `sample_yards_gained(field_pos: float, down: int, zone_params: dict) -> Tuple[int, bool]`
The core sampling function that drives the Monte Carlo simulation:

1. **Determine field zone**: Maps field position to one of four zones
2. **Get play probabilities**: Retrieves probabilities for run/pass/sack outcomes
3. **Hierarchical sampling**:
   - First samples play type (run vs. pass vs. sack)
   - Then samples yards from the appropriate distribution
4. **Turnover simulation**: Samples for interceptions (passes) and fumbles (runs), returning both yards and turnover status

**Distribution Types**:
- **Runs**: Mixture model with separate exponential distributions for positive and negative outcomes
- **Short passes**: Normal distribution (most predictable gains)
- **Deep passes**: Gamma distribution (high variance, right-skewed)
- **Sacks**: Negative exponential distribution (always lose yards)

**Turnover Modeling**:
- **Interceptions**: Sampled for passing plays (short and deep pass) using interception rates by field zone
- **Fumbles**: Sampled for rushing plays using fumble rates by field zone
- Returns tuple of (yards_gained, is_turnover) for use in drive simulation

#### `_sample_from_exponential_mixture(dist: PlayTypeDistribution, positive: bool) -> int`
Implements mixture model sampling for runs:
- Positive runs: Standard exponential with rate parameter 1/mean
- Negative runs: Negative exponential (fumbles, TFLs)

#### `_get_field_zone(field_pos: float) -> str`
Simple utility that maps field position to zone name based on FIELD_ZONES constants.

### 5. Expected Points Calculation

#### `ExpectedPointsCalculator` Class
The main orchestrator class that coordinates all modeling components.

**Initialization** (`__init__`):
1. Fits play distribution parameters for each zone
2. Fits field goal logistic regression
3. Calculates average punt distances by zone

These are computed once during initialization for efficiency.

##### Key Methods:

**`_fit_zone_parameters() -> dict`**
- Iterates through each field zone (own, midfield, fg_range, red_zone)
- Calls `_fit_play_distribution()` for each play type (run, short_pass, deep_pass, sack)
- Calls `_fit_turnover_rates()` for each play type to get interception and fumble rates
- Returns dictionary mapping zone names to FieldZoneParams objects

**`_fit_play_distribution(zone_data: pd.DataFrame, play_type: str) -> PlayTypeDistribution`**
- Filters data for specific play type in zone
- Calculates probability, mean, and standard deviation
- For deep passes, computes alpha/beta parameters for gamma distribution:
  - `alpha = mean² / variance`
  - `beta = mean / variance`

**`_fit_turnover_rates(zone_data: pd.DataFrame, play_type: str) -> TurnoverRates`**
- Filters data for specific play type in zone
- Calculates interception rate: (number of interceptions) / (total passes)
- Calculates fumble rate: (number of fumbles) / (total plays of type)
- Returns TurnoverRates with both probabilities

**`calculate_expected_points(down, field_pos, yards_to_go, num_simulations) -> float`**
The main public API:
1. Runs multiple drive simulations from the given game state
2. Averages the results to get expected points
3. Returns a single float value representing expected points

**`_simulate_drive(down, field_pos, yards_to_go, possession) -> float`**
Recursive function that simulates a complete drive until scoring:

1. **Base Cases** (Scoring events):
   - Field position ≥ 100: Touchdown (±7 points)
   - Field position ≤ 0: Safety (±2 points)

2. **Non-Fourth Down Logic** (down 1-3):
   - Sample yards gained and check for turnover from zone parameters
   - If turnover: flip possession and field position, restart at new position
   - Check for scoring outcomes
   - Determine if first down achieved
   - Recursively continue drive with updated state

3. **Fourth Down Decision Logic** (down 4):
   - **Yards to go > 2**: Attempt field goal if in range (≥65), otherwise punt
   - **Yards to go ≤ 2**: 
     - Punt if on own side (≤40)
     - Go for it if closer to goal line
     - Sample for turnover during go-for-it attempt
     - Turnover on downs if not achieved

**`_attempt_field_goal(field_pos, possession) -> float`**
- Samples make/miss using field goal success probability
- Returns points if made
- Handles missed FG as turnover with field position flip

**`_punt(field_pos, possession) -> float`**
- Looks up zone-specific punt distance
- Updates field position and possession
- Continues simulation from new state

## Usage Pattern

```python
# 1. Load and prepare data
df = load_and_prepare_data("path/to/2019_pff_all_plays.csv")

# 2. Create calculator (fits all distributions)
calc = ExpectedPointsCalculator(df)

# 3. Query expected points
ep = calc.calculate_expected_points(
    down=1,
    field_pos=25,
    yards_to_go=10,
    num_simulations=100
)
```

## Key Design Decisions

### Monte Carlo Simulation
Rather than calculating expected value analytically, the model simulates complete drives and averages outcomes. This allows for:
- Complex conditional logic (4th down decisions)
- Non-linear relationships
- Easy validation against actual game data

### Field Zone Segmentation
Different field positions have different offensive characteristics:
- **Own side**: Conservative play-calling, longer drives
- **Midfield**: Balanced play-calling
- **FG Range**: More aggressive passing
- **Red Zone**: Most aggressive, highest TD probability

### Logistic Regression for Field Goals
Logistic regression naturally models the probability constraint (0-1) and captures the sigmoidal relationship between distance and success.

### Recursive Drive Simulation
Recursion cleanly models the sequential nature of plays while naturally handling complex scenarios:
- Turnovers reset state
- Field position flipping
- Down/distance tracking

## Validation Points

The model can be validated by:
1. Comparing simulated EP values to empirical data (EPA)
2. Checking that EP decreases as you move away from opponent's endzone
3. Validating that 4th down decisions align with real coaching behavior
4. Comparing to published expected points models (e.g., nflscrapy)

## Limitations & Future Improvements

### Current Limitations
- Doesn't model two-point conversions after touchdowns
- Doesn't account for fumble recovery outcomes (assumes defensive possession)
- Field zone parameters are static (could vary by team/era)
- Doesn't model time constraints or game clock
- Turnover rates are aggregated across all plays (could vary by down/distance)

### Potential Improvements
- Implement two-point conversion logic (based on success rate by distance)
- Model fumble recovery probability (not all fumbles are lost)
- Fit separate parameters for different play-calling tendencies by team
- Add timeout and clock management
- Optimize recursion (memoization) for faster computation
- Cross-validate parameters using hold-out test set
- Condition turnover rates on down and yards to go
- Model defensive/special teams scoring (pick-sixes, defensive TDs)

## Dependencies

- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing and random sampling
- `scipy.stats`: Statistical distributions
- `scikit-learn`: Logistic regression modeling
- `dataclasses`: Type-safe data structures
- `typing`: Type hints for better code documentation
- `logging`: Event tracking and debugging

All are standard Python data science libraries available via pip.
