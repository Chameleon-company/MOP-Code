# **Traffic Congestion Recommendations System Based On Climate Condition**

**Authored by:** Quang An Quoc Tran  

**Student ID:** 224696086

**Duration:** 360 mins  

**Level:** Intermediate  

**Pre-requisite Skills:** Python, Data wrangling, Machine Learning, Large Language Model, Prompt Engineering, Retrieval Augmented Generation

## Introduction
Urban traffic congestion is a persistent challenge in modern cities, affecting mobility, economic productivity, and environmental sustainability. Traditional traffic analysis often relies primarily on historical movement patterns, which may not fully capture the dynamic factors influencing congestion. With the emergence of microclimate sensor data, it is now possible to incorporate environmental conditions such as temperature, humidity, wind, air quality, and noise into traffic analysis.

This project develops a Traffic Congestion Recommendation System by integrating large-scale transport activity data with microclimate data to better understand and predict congestion patterns. The transport dataset provides high-frequency records of urban mobility across multiple locations, while the environmental dataset adds contextual information that may influence traffic behavior.

By combining these data sources, the system applies data preprocessing, feature engineering, and machine learning techniques to transform raw data into meaningful insights. Predictive models such as Logistic Regression and Random Forest are used to classify congestion levels, enabling more informed decision-making for traffic management and urban planning.

## Scenario
In a smart city context, both commuters and transportation authorities require accurate and timely insights into traffic conditions. Rather than relying solely on past traffic trends, this system leverages both temporal patterns and environmental conditions to provide more intelligent predictions of congestion.

For instance, congestion levels often follow clear daily cycles, with higher traffic during morning and evening peak hours and lower activity during midday. However, these patterns can be influenced by environmental factors such as temperature, humidity, or wind conditions. By integrating these variables, the system can better capture variations in traffic behavior across different locations and times.

In practice, the system can predict whether a specific area is likely to be congested at a given time and support recommendations such as optimal travel times or alternative routes. Additionally, it provides valuable insights for city planners to understand how environmental conditions interact with urban mobility, ultimately contributing to more efficient and sustainable transportation systems.

## Dataset Description
The project utilizes two primary datasets: a transport activity dataset and a microclimate environmental dataset. The transport dataset contains more than 5.4 million high-frequency traffic activity records collected from multiple urban monitoring locations throughout Melbourne. Each record includes the location identifier, geographic coordinates, timestamp, transport class, and movement count. The transport classes include pedestrians, cyclists, cars, taxis, buses, vans, trucks, and several other mobility categories, providing a detailed representation of urban traffic behavior over time.

The microclimate dataset contains approximately 347 thousand environmental sensor observations collected across distributed monitoring stations. The dataset includes important atmospheric and environmental variables such as air temperature, relative humidity, atmospheric pressure, wind speed, wind direction, PM2.5, PM10, and noise levels. These environmental factors provide additional context that may influence traffic movement and congestion patterns.

The integration of both datasets enables the development of a richer analytical framework where transportation activity can be studied together with environmental conditions. This combined dataset supports advanced congestion prediction and intelligent traffic recommendation systems.

## Data Preprocessing
Data preprocessing plays an important role in ensuring the quality, consistency, and reliability of the dataset before machine learning modeling. The first preprocessing step involved converting all datetime fields into a standardized timezone format using Melbourne local time. This ensured temporal consistency between the transport and environmental datasets.

Because the transport and microclimate datasets were collected independently, spatial and temporal alignment was required before merging. The transport activity data was aggregated into one-hour intervals, while the environmental data was also grouped into hourly averages. Spatial matching between transport locations and environmental sensors was performed using the BallTree nearest-neighbor algorithm with Haversine distance calculations. Only sensors located within one kilometer of transport locations were retained to ensure meaningful environmental associations.

After merging the datasets, additional preprocessing steps were performed to improve data quality. Missing values were identified primarily in PM2.5, PM10, Noise, and GustWindSpeed features. These missing values were handled using linear interpolation and median imputation techniques. Outlier handling was also applied using the Interquartile Range (IQR) method to reduce the impact of extreme observations. Finally, cyclical temporal features such as `hour_sin` and `hour_cos` were created to better represent repeating daily traffic patterns.

## Feature Engineering
Feature engineering was applied to transform raw transport activity into a meaningful congestion prediction problem. Since different transportation classes contribute differently to traffic congestion, weighted scores were assigned to each transport type. For example, heavier vehicles such as buses and trucks received larger weights than pedestrians and cyclists. These weighted counts were aggregated to generate a congestion score for each location and hour.

An adaptive thresholding approach was then used to convert congestion scores into binary labels. The 70th percentile congestion value for each location was selected as the congestion threshold. Observations above this threshold were labeled as congested, while observations below the threshold were labeled as non-congested. This location-specific thresholding method allows congestion classification to adapt to different traffic characteristics across monitoring locations.

Additional temporal features such as hour of day, day of week, weekend indicators, and cyclical encodings were included to capture daily traffic patterns. Environmental features including temperature, humidity, wind conditions, atmospheric pressure, pollution levels, and noise were also retained as predictive variables.

## Exploratory Data Analysis
Exploratory Data Analysis was conducted to better understand the distribution of congestion patterns and feature relationships. The descriptive statistics revealed that the congestion score distribution was highly right-skewed, indicating the presence of peak congestion events during certain periods. Congestion levels varied significantly across locations and times, reflecting the complexity of urban mobility behavior.

Temporal analysis showed that congestion followed a strong daily cycle, with traffic peaks occurring during morning and evening commuting periods. Midday periods generally experienced lower congestion levels. Correlation analysis indicated that temporal features had the strongest relationship with congestion, while environmental variables exhibited weaker but still meaningful relationships.

Feature importance analysis using Random Forest demonstrated that cyclical time features such as `hour_cos` and `hour_sin` were the most influential predictors. Environmental variables such as air temperature, humidity, atmospheric pressure, and wind conditions also contributed to prediction performance, suggesting that weather conditions partially influence traffic dynamics.

## Machine Learning Models
Several machine learning models were developed and evaluated to predict traffic congestion conditions. Logistic Regression was first implemented as a baseline classification model. Although the model achieved moderate performance, its linear assumptions limited its ability to capture complex interactions between environmental and temporal variables.

Random Forest significantly improved predictive performance by capturing non-linear relationships within the dataset. The model achieved an accuracy of 77.12% and a ROC-AUC score of 0.8240. Random Forest also demonstrated balanced precision and recall, making it highly effective for congestion classification.

XGBoost further improved prediction performance through gradient boosting techniques and advanced regularization. The optimized XGBoost model achieved an accuracy of 72.30%, recall of 82.02%, F1-score of 0.6054, and ROC-AUC score of 0.8273 after fine-tuning. The high recall performance indicates that the model was particularly effective at identifying congestion events, which is important for intelligent traffic management applications.

After comparing all models, XGBoost was selected as the final congestion prediction model due to its superior recall and overall predictive performance.

## Model Fine-Tuning
Model fine-tuning was performed using `RandomizedSearchCV` combined with `TimeSeriesSplit` cross-validation to preserve the temporal structure of the dataset. Multiple hyperparameters were optimized, including tree depth, learning rate, number of estimators, feature sampling ratios, and regularization parameters.

The optimized Random Forest model achieved a cross-validation F1-score of 0.6726 and produced strong test performance with an accuracy of 73.59% and ROC-AUC score of 0.8267. The optimized XGBoost model achieved a cross-validation F1-score of 0.6572 and a test ROC-AUC score of 0.8273.

Threshold optimization was also applied to improve the balance between precision and recall. Instead of relying solely on the default probability threshold of 0.50, the best classification threshold was selected using F1-score maximization. This process improved congestion detection performance while reducing false predictions.

## LLM Explanation System

To improve model interpretability, a Large Language Model explanation system was integrated into the project. The purpose of this component is to translate machine learning predictions into human-readable explanations and actionable traffic insights. Instead of displaying only numerical predictions, the system explains why congestion is expected based on environmental and temporal factors.

The project evaluated multiple lightweight language models including Tiny Llama 1B and Gemini 3.1 Flash Lite Preview. Tiny Llama 1B achieved an overall explanation score of 0.6852 with a feature coverage score of 0.6671. While the model successfully explained several important variables, some environmental and engineered temporal features were omitted, resulting in lower explanation completeness and weaker contextual reasoning.

Gemini 3.1 Flash Lite Preview produced more coherent, structured, and context-aware explanations compared to Tiny Llama 1B. Before optimization, the model achieved an overall explanation score of 0.7882 with a feature coverage score of 0.5294. Although Gemini mentioned fewer total features, it generated more natural and human-readable explanations with stronger contextual understanding, full prediction consistency, and 100% location matching accuracy. The model also provided more practical traffic insights and recommendation-oriented responses, making it more suitable for explainable AI applications in intelligent transportation systems.

## System Optimization
System optimization focused on improving both machine learning performance and the quality of the LLM explanation system through prompt engineering and Retrieval-Augmented Generation (RAG). Prompt engineering was used to guide the LLM in generating structured and consistent explanations by explicitly instructing the model to describe congestion predictions, environmental conditions, temporal patterns, and practical traffic recommendations. Through iterative prompt refinement, the generated responses became more coherent, contextual, and aligned with the machine learning prediction outputs.

RAG was implemented to improve explanation accuracy and reduce hallucination issues by retrieving real prediction results, feature values, and traffic labels directly from the machine learning pipeline before generating responses. This ensured that the explanations remained grounded in actual congestion prediction data instead of relying solely on the LLM’s internal knowledge. After optimization, the Gemini 3.1 Flash Lite Preview model achieved a perfect overall explanation score of 1.0 and a feature coverage score of 1.0, successfully explaining all important traffic, temporal, spatial, and environmental features while maintaining full prediction consistency and 100% location matching accuracy.

## Conclusion
This project successfully developed a Traffic Congestion Recommendation System by integrating large-scale transport activity data with environmental microclimate sensor data. Through data preprocessing, feature engineering, machine learning modeling, and explainable AI integration, the system demonstrated strong capability in predicting congestion conditions under varying environmental and temporal contexts.

The results showed that temporal features were the most influential predictors of congestion, while environmental variables such as temperature, humidity, wind conditions, and pollution levels also contributed meaningful predictive value. Among the evaluated models, XGBoost achieved the best overall performance and was selected as the final prediction model.

The integration of Large Language Models further improved system interpretability by generating natural language explanations and traffic recommendations. Overall, this project demonstrates how machine learning and environmental sensing technologies can support smarter and more sustainable urban transportation systems.

## Future Improvements
Future work could further improve the system by incorporating real-time streaming traffic data, weather forecast APIs, and GPS trajectory information. Deep learning architectures such as LSTM or Transformer-based time-series models may also improve predictive performance by capturing long-term temporal dependencies.

Additional improvements could include route recommendation systems, live congestion visualization dashboards, and integration with smart traffic signal management systems. Expanding the geographical coverage of environmental sensors and transportation monitoring infrastructure would also improve prediction reliability and scalability for larger smart city deployments.