# CQ-Data-Track-2026
This project focuses on the data track for Code Quantum 2026. The idea for this project is to implement machine learning methods with a data science approach with the theme for this hackathon being F1.

The project that I created uses a database provided by the Code Quantum resources based around the theme of F1 drivers. The approach I took for this hackathon was machine learning approach using a Random Forest Classifier to determine what actually constitutes an "outperformance" in a race. It analyzes four key features: Pace Delta, Grid Position, Lap Standard Deviation and Average Lap Time. The core of the project is the Machine Learning Driver Score. Instead of arbitrarily picking weights for these stats, the program: Trains a model to predict if a driver will "outperform" based on historical race data. Extracts the Feature Importance (the weights the AI assigned to each stat). Calculates a final score using the formula:

$$\text{ML\_Score} = \sum_{i=1}^{n} (f_i \cdot w_i \cdot 1)$$

Where:
- $f_i$: The value of the specific feature
- $w_i$: The assigned weight for that feature
- $n$: The total number of features

The project generates several data-driven rankings:
Consistency Index: Measures a driver's ability to maintain steady lap times while gaining positions.
Clutch Factor: Highlights drivers who significantly improve their position from their starting grid slot.
ML Score: The definitive ranking based on AI-weighted performance metrics.

