# College Football Expected Points Model

## Overview
This repository contains an advanced statistical model for predicting expected points in college football, providing valuable insights into play-by-play game situations. By leveraging recursive simulations and Bayesian statistical methods, the project helps evaluate strategic decisions and optimize coaching tactics. I used college football play-by-play data from 2019. 

## Motivation
Understanding expected points at different field positions allows teams to make data-driven decisions that can significantly impact game outcomes. This model assists in analyzing the trade-offs between various play calls—such as going for a field goal versus a punt—and supports more informed coaching strategies.

## Key Features
- **Recursive Simulation**: Simulates play sequences to estimate expected points across various field positions and game scenarios.
- **Bayesian Modeling**: Utilizes semi-Bayesian methodology by sampling yardage gains and uncertainties, enriching the predictive power of the model.
- **Logistic Regression for Field Goals**: Provides insight into field goal success probabilities to aid in decision-making between punting, kicking, or advancing the ball.

## Technologies Used
- **Programming Language**: R
- **Libraries**:
  - `tidyverse` for data manipulation and visualization
  - `ggplot2` for data visualization
- **Statistical Techniques**:
  - Bayesian sampling
  - Logistic regression
  - Recursive modeling

## Further details on methods and improvements
- *Logistic regression*: Created a glm logistic model using past kick distance and outcomes to simulate kick success probability from coin flip sampling. The probability of success was generated from the linear model with field position as the input.

- *Recursive play simulation (main function)*: Considering different scenarios based on field position, down, distance, etc., I calculated the expected points from that given position by recursively adjusting the game state until a scoring event occurred by either team. Later, I would use this to simulate many observations from the same situation and average the points scored (negative for opponent score) to get the expected points.

- *Matching statistical distributions to play type outcomes*: For non-special teams plays, I categorized possible results into run, sack, short pass, and deep pass. For each of these, I plotted the distribution of resulting yards and mapped each to a corresponding distribution. The yards outcome for each was tweaked slightly depending on where they were on the field (own side, near midfield, field goal range, red zone)
  - Run: mixture model of two exponential distributions - one for negative runs and one for positive runs (sampled probability of positive/negative outcome before analyzing actual yards gained)
  - Sack: negative exponential distribution
  - Short pass: normal distribution
  - Deep pass: gamma distribution

- *Visualizing yards gained distributions*: Used simple plotting in R to visualize expected yards distributions after many runs of each play type.

- *Things that could be improved in future runs*
  - Efficiency: There was a lot of copy and paste code, especially for analyzing the unique play distributions and storing each individually, that could be performed better using functions.
  - Code documentation: While there are some comments that help understand my thought process, I feel there is always room for improvement on documentation
  - Code structure: more thought could have been put into code structure prior to beginning coding, sometimes had to backtrack to help efficiency
  - Sampling speed: could have stored many samples to be iterated through when needed instead of sampling each play
