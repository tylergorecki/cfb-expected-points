# College Football Expected Points Model

## Overview
This project develops a comprehensive expected points model for college football using recursive Monte Carlo simulations and statistical modeling of play outcomes. The model estimates the expected points a team will score from any given field position by simulating complete drives under various game scenarios. Built using 2019 PFF college football play-by-play data, it provides analytical insights for evaluating fourth-down decisions, field goal attempts, and punting strategies. 

## Motivation
Understanding expected points at different field positions allows teams to make data-driven decisions that can significantly impact game outcomes. This model assists in analyzing the trade-offs between various play calls—such as going for a field goal versus a punt—and supports more informed coaching strategies.

## Key Features
- **Recursive Monte Carlo Simulations**: Recursively simulates complete drives from any field position, modeling all possible play outcomes (runs, passes, sacks, punts, field goals, turnovers) until a scoring event occurs.
- **Probabilistic Yardage Modeling**: Uses statistical distributions fitted to actual play data to model yards gained by play type (runs, short passes, deep passes, sacks).
- **Field Position Context**: Adjusts play probabilities and outcome distributions based on four field position zones (own side, midfield, field goal range, red zone).
- **Logistic Regression for Field Goals**: Employs logistic regression on historical kick data to model field goal success probability by distance.

## Technologies Used
- **Programming Language**: R
- **Libraries**:
  - `tidyverse` for data manipulation and visualization
  - `ggplot2` for data visualization
- **Statistical Techniques**:
  - Bayesian sampling
  - Logistic regression
  - Recursive modeling
## Methodology

### 1. Data Preparation
- Extracted field goal attempts from 2019 PFF data, standardizing field position encoding
- Classified all non-special-teams plays into categories: runs, short passes, deep passes, and sacks
- Segmented field into four zones for localized analysis: Own Side (≤40 yards), Midfield (40-65 yards), Field Goal Range (65-80 yards), Red Zone (≥80 yards)

### 2. Yardage Distribution Modeling
Analyzed actual play outcomes and fitted statistical distributions to each play type:
- **Runs**: Mixture model of two exponential distributions (separate positive and negative outcome distributions)
- **Sacks**: Negative exponential distribution  
- **Short Passes**: Normal distribution
- **Deep Passes**: Gamma distribution
- **Punts**: Field position-dependent punt distance analysis

Each distribution's parameters vary by field position zone to reflect realistic situational play outcomes.

### 3. Play Type Probability Modeling
Calculated conditional probabilities for each play type within each field position zone:
- Run probability vs. Pass probability (run, short pass, deep pass, sack)
- Used these to weight the play selection during simulations

### 4. Field Goal Success Modeling
- Built logistic regression model using historical field goal data
- Model predicts success probability as a function of field position
- Used coin flip sampling to stochastically determine make/miss outcomes

### 5. Recursive Expected Points Calculation
The core `expoints()` function:
1. Takes game state (down, field position, yards to go, possession) as input
2. Samples yards gained from appropriate distribution based on play type probabilities
3. Updates field position and evaluates game state:
   - **Scoring outcomes**: Touchdowns (+7/-7), Safeties (+2/-2), Field Goals (+3/-3)
   - **Special situations on 4th down**: Decides between kicking (if in range), punting, or going for it
   - **Turnover on downs**: Flips possession and field position
4. Recursively continues simulation until scoring event
5. Returns expected points for the initial game state

### 6. Validation and Visualization
- Plotted actual yard distributions to verify fit quality
- Summarized play-by-play statistics (average yards gained, play frequencies by field zone)

## Future Improvements
- **Code Organization**: Refactor repetitive distribution setup code into generalized functions
- **Documentation**: Expand inline comments and add function documentation strings
- **Efficiency**: Pre-sample and store distributions rather than sampling each play individually
- **Architecture**: Better upfront planning of code structure to minimize refactoring
- **Validation**: Compare model outputs against empirical expected points data from actual game outcomeso backtrack to help efficiency
  - Sampling speed: could have stored many samples to be iterated through when needed instead of sampling each play
