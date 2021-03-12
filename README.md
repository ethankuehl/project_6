# Predicting the Next Pitch: Utilizing Machine Learning to Predict the Next Pitch Thrown

## Problem Statement

MLB scouting departments have become heavily reliant on data analysis since the advent of Moneyball. With the added data being provided by Statcast - a high-speed, high-accuracy, automated tool developed to analyze player movements and athletic abilities in Major League Baseball - teams are now looking to get that extra edge from Machine Learning. The goal of this project is to utilize statcast data and machine learning algorithms from the 2018 MLB season to predict a pitcher's next pitch based on the pitcher's repertoire and the situational context under which the pitch is being thrown. Furthermore, this predictive modeling could be used to identify undervalued players as potential trade targets for the upcoming 2021 season. 

## Background
The 2017 Astros will always have an asterisk next to their World Series title due to a cheating scandal that occurred throught the regular and postseason. Their scheme involved have a well-placed camera zoomed in on the catcher's hand as he relayed the type of pitch he wanted the pitcher to throw next. A live-stream of this camera was placed in the dugout where a technician would bang a trash can to indicate whether a fastball or off-speed pitch was coming next. 

While I am not necessarily motivated by the Astros scoundrelous ways, this scandal did inspire me to explore whether there is a better way to prepare hitters for certain situations against certain types of pitchers. Furthermore, I became curious if one could become even more granular with their predictions and determine whether a fastball, off-speed (change-up), or breaking ball (slider, curveball) is coming next. 

One thing that became apparent quickly: predicting pitches is quite difficult. Pitchers are intentionally random to throw off the hitters timing and gain the upperhand. However, patterns did emerge in certain situations that might be helpful to hitters in their preparation. As one of the all-time greats Hank Aaron once said: **“Guessing what the pitcher is going to throw is 80 percent of being a successful hitter. The other 20 percent is just execution.”**

## Executive Summary
An executive summary:
### Goal
The goal of this study is to predict the next pitch based on the pitcher's repertoire, hitter's previous season batting average, and the situational context under which the pitch is being thrown. 

### Data Collection
I downloaded data from Baseball Savant's Statcast website utilizing the baseball_scraper python package. 


What are your metrics?
What were your findings?
What risks/limitations/assumptions affect these findings?
Summarize your statistical analysis, including:
implementation
evaluation
inference
