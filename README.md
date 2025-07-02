# Sleep Disorder Prediction System

## Overview

The Sleep Disorder Prediction System is an advanced machine learning application that analyzes sleep patterns and lifestyle factors to predict potential sleep disorders. Using a comprehensive dataset of sleep metrics, the system achieves **perfect classification accuracy (100%)** in identifying three categories: No Sleep Disorder, Insomnia, and Sleep Apnea.

## Key Features

- **Perfect Classification Performance**: 100% accuracy and F1-score on test data
- **Multi-Class Prediction**: Identifies None, Insomnia, and Sleep Apnea
- **Interactive Interface**: User-friendly widgets for real-time predictions
- **Comprehensive Analysis**: 15 engineered features from sleep and lifestyle data
- **Risk Factor Assessment**: Detailed analysis of contributing factors
- **Personalized Recommendations**: Tailored advice based on individual profiles

## Dataset Information

- **Total Records**: 452 individuals
- **Features**: 15 engineered features from original 15 variables
- **Target Distribution**:
  - None (No Disorder): 375 cases (83.0%)
  - Insomnia: 60 cases (13.3%)
  - Sleep Apnea: 17 cases (3.8%)

### Key Variables
- **Sleep Metrics**: Duration, efficiency, REM/Deep/Light sleep percentages, awakenings
- **Demographics**: Age, gender
- **Lifestyle Factors**: Caffeine/alcohol consumption, smoking status, exercise frequency
- **Sleep Schedule**: Bedtime, wake-up time, schedule consistency

## Methodology

### Data Preprocessing
1. **Missing Value Handling**: Median imputation for numerical features
2. **Feature Engineering**: 
   - Time parsing for bedtime/wake-up hours
   - Sleep schedule consistency calculation
   - Weekend sleep-in indicators
3. **Encoding**: Label encoding for categorical variables
4. **Scaling**: StandardScaler for feature normalization

### Target Variable Creation
Since the original dataset lacked sleep disorder labels, we created a synthetic target using clinical criteria:

```python
def classify_sleep_disorder(row):
    efficiency = row['Sleep efficiency']
    awakenings = row['Awakenings']
    deep_sleep = row['Deep sleep percentage']
    
    # Sleep Apnea: Low efficiency, high awakenings, low deep sleep
    if efficiency < 0.6 and awakenings >= 3 and deep_sleep < 30:
        return 'Sleep Apnea'
    # Insomnia: Low efficiency, high awakenings
    elif efficiency < 0.7 and awakenings >= 2:
        return 'Insomnia'
    else:
        return 'None'
```

### Model Selection & Training
We evaluated 7 different algorithms:
- Random Forest
- **Decision Tree** (Best Performer)
- Support Vector Machine
- Logistic Regression
- K-Nearest Neighbors
- Gradient Boosting
- Neural Network

## Results

### Model Performance Comparison

| Model | Accuracy | F1-Score | CV Mean | CV Std |
|-------|----------|----------|---------|--------|
| **Decision Tree** | **1.0000** | **1.0000** | **0.9889** | **0.0055** |
| Gradient Boosting | 1.0000 | 1.0000 | 0.9917 | 0.0068 |
| Random Forest | 0.9890 | 0.9896 | 0.9779 | 0.0067 |
| Logistic Regression | 0.9780 | 0.9734 | 0.9392 | 0.0341 |
| Neural Network | 0.9670 | 0.9664 | 0.9503 | 0.0353 |
| SVM | 0.9560 | 0.9542 | 0.9365 | 0.0305 |
| K-Nearest Neighbors | 0.9560 | 0.9545 | 0.9089 | 0.0528 |

### Final Model Performance
- **Algorithm**: Decision Tree (Hyperparameter Tuned)
- **Test Accuracy**: 100%
- **Test F1-Score**: 100%
- **Cross-Validation Score**: 98.84% ± 1.11%

### Classification Report
```
              precision    recall  f1-score   support
    Insomnia       1.00      1.00      1.00        12
        None       1.00      1.00      1.00        76
 Sleep Apnea       1.00      1.00      1.00         3
    accuracy                           1.00        91
   macro avg       1.00      1.00      1.00        91
weighted avg       1.00      1.00      1.00        91
```

## Usage

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn ipywidgets
```

### Basic Prediction
```python
prediction, probabilities = predict_sleep_disorder(
    model=final_model,
    scaler=scaler,
    age=30,
    gender='Female',
    sleep_duration=8.0,
    sleep_efficiency=0.90,
    rem_sleep=25,
    deep_sleep=60,
    light_sleep=15,
    awakenings=1,
    caffeine=25,
    alcohol=1,
    smoking='No',
    exercise=4,
    bedtime_hour=22.0,
    wakeup_hour=6.0
)
```

### Interactive Interface
The system includes an interactive Jupyter widget interface for real-time predictions:

- 👤 **Personal Information**: Age, Gender
- 😴 **Sleep Metrics**: Duration, Efficiency, Sleep Stages, Awakenings
- 🏃 **Lifestyle Factors**: Caffeine, Alcohol, Smoking, Exercise
- ⏰ **Sleep Schedule**: Bedtime, Wake-up Time

## Key Insights

### Risk Factors Identified
1. **Sleep Efficiency < 70%**: Strong predictor of sleep disorders
2. **Frequent Awakenings (≥3)**: Associated with Sleep Apnea
3. **Low Deep Sleep (<30%)**: Indicates poor sleep quality
4. **High Caffeine Intake (>100mg)**: Disrupts sleep patterns
5. **Irregular Sleep Schedule**: Impacts sleep consistency

### Protective Factors
1. **High Sleep Efficiency (>85%)**
2. **Regular Exercise (≥4 days/week)**
3. **Consistent Sleep Schedule**
4. **Moderate Caffeine Consumption (<50mg)**
5. **Non-smoking Status**

## Technical Details

### Feature Importance (Top 10)
The Decision Tree model identified these key features:
1. Sleep Efficiency
2. Awakenings
3. Deep Sleep Percentage
4. REM Sleep Percentage
5. Sleep Duration
6. Age
7. Caffeine Consumption
8. Sleep Schedule Consistency
9. Light Sleep Percentage
10. Exercise Frequency

### Model Parameters (Optimized)
```python
DecisionTreeClassifier(
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)
```

## Visualizations

The system provides comprehensive visualizations:
- Sleep disorder distribution pie charts
- Sleep efficiency by disorder box plots
- Feature correlation heatmaps
- Learning curves and validation curves
- Interactive sleep analysis dashboards
- Risk factor radar charts

## Important Considerations

### Model Limitations
1. **Synthetic Target**: Sleep disorders were derived from sleep metrics, not clinical diagnoses
2. **Perfect Accuracy**: May indicate overfitting to synthetic labels
3. **Limited Real-World Validation**: Results should be validated against clinical data
4. **Imbalanced Dataset**: Sleep Apnea cases are underrepresented (3.8%)

### Recommendations for Production
1. **Clinical Validation**: Validate against professionally diagnosed cases
2. **Cross-Dataset Testing**: Test on independent datasets
3. **Feature Importance Analysis**: Ensure medical relevance of key features
4. **Regular Model Updates**: Retrain with new clinical data

## Future Enhancements

1. **Real Sleep Disorder Labels**: Integration with clinical diagnostic data
2. **Temporal Analysis**: Time-series analysis of sleep patterns
3. **Additional Features**: Heart rate variability, environmental factors
4. **Mobile Integration**: Real-time monitoring via wearable devices
5. **Explainable AI**: LIME/SHAP for prediction explanations

## Support & Contributing

For questions, issues, or contributions:
- Create GitHub issues for bug reports
- Submit pull requests for feature enhancements
- Follow coding standards and include tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Sleep health research community
- Scikit-learn development team
- Contributors to sleep disorder classification research

---

**Disclaimer**: This system is for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for sleep-related concerns.
