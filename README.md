# College football expected points project

Used college football play-by-play data from 2019 to create a function that predicts the number of expected points based on field position, down, and yards to go. The pdf is a paper we wrote for our class answering the question of when, in college football, it is better to punt as opposed to going for it on 4th down. 

## Methods used
- *Logistic regression*: Created a glm logistic model using past kick distance and outcomes to simulate kick success probability from coin flip sampling. The probability of success was generated from the linear model with field position as the input.

- 
