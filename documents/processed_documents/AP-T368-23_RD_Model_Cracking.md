

## Page 1
 
 
Technical Report 
AP-T368-23 
Austroads Road Deterioration Model Update 
Cracking 


## Page 2
Austroads Road Deterioration Model Update: Cracking  
Prepared by 
Dr Tim Martin and Ranita Sen 
 
Publisher 
Austroads Ltd. 
Level 9, 570 George Street 
Sydney NSW 2000 Australia 
Phone: +61 2 8265 3300 
austroads@austroads.com.au 
www.austroads.com.au 
 
Project Manager 
Dr Zahid Hoque 
 
Abstract 
The aim of the project was to update the current Austroads road 
deterioration (RD) models using the long-term pavement 
performance and long-term pavement performance maintenance 
(LTPP/LTPPM) dataset and other data available, such as the traffic 
sped deflectometer (TSD) datasets, to improve these models’ 
explanatory power.  
The cracking RD model documented in this report was based on a 
mechanistic-empirical, deterministic approach using a multi-variate 
non-linear regression (MVNLR) analysis. The cracking RD model 
uses cumulative cracking as the dependent variable for thinly 
surfaced flexible unbound granular pavements. The model was 
developed using the New South Wales dataset collected by the TSD 
from 2014 to 2018. The cracking RD model will need calibration to 
suit locally observed cracking conditions as location has a bearing on 
the rates of cracking progression.  
 
About Austroads 
Austroads is the peak organisation of Australasian road 
transport and traffic agencies. 
Austroads’ purpose is to support our member organisations to 
deliver an improved Australasian road transport network. To 
succeed in this task, we undertake leading-edge road and 
transport research which underpins our input to policy 
development and published guidance on the design, 
construction and management of the road network and its 
associated infrastructure.  
Austroads provides a collective approach that delivers value 
for money, encourages shared knowledge and drives 
consistency for road users. 
Austroads is governed by a Board consisting of senior 
executive representatives from each of its eleven member 
organisations:  
• Transport for NSW 
• Department of Transport and Planning Victoria 
• Queensland Department of Transport and Main Roads 
• Main Roads Western Australia  
• Department for Infrastructure and Transport South Australia  
• Department of State Growth Tasmania  
• Department of Infrastructure, Planning and Logistics 
Northern Territory  
• Transport Canberra and City Services Directorate, 
Australian Capital Territory  
• Department of Infrastructure, Transport, Regional 
Development, Communications and the Arts 
• Australian Local Government Association  
 
Keywords 
prediction of cracking progression, flexible thin surfaced unbound 
granular pavements,  
 
ISBN 978-1-922700-68-1 
Austroads Project No. AAM6214  
Austroads Publication No. AP-T368-23 
Publication date January 2023 
Pages 33 
© Austroads 2023 
This work is copyright. Apart from any use as permitted under the 
Copyright Act 1968, no part may be reproduced by any process 
without the prior written permission of Austroads. 
Acknowledgements 
The authors wish to acknowledge the support they received from their independent technical advisors; Dr Yongze Song (Curtin 
University), Dr Theuns Henning (Infrastructure Decision Support, NZ) and Professor Jayantha Kodikara (Director ARC Smart Hub, 
Monash University). In addition, the authors acknowledge the support received from all the contributing road agencies from the Australian 
states and territories and New Zealand through the Project Working Group including the supply of data. The authors also acknowledge the 
technical support of their colleagues Tyrone Toole, Lith Choummanivong and Thorolf Thoresen, and gratefully acknowledge ARRB survey 
and data processing teams across Australia. 
This report has been prepared for Austroads as part of its work to promote improved Australian and New Zealand transport outcomes by 
providing expert technical input on road and road transport issues. 
Individual road agencies will determine their response to this report following consideration of their legislative or administrative 
arrangements, available funding, as well as local circumstances and priorities. 
Austroads believes this publication to be correct at the time of printing and does not accept responsibility for any consequences arising from 
the use of information herein. Readers should rely on their own skill and judgement to apply information to particular issues. 


## Page 3
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page i 
Summary 
Road deterioration (RD) models help in predicting future road condition and assessing maintenance needs. 
The current Austroads RD models available for use by road agencies and industry were developed in 2008 
using observational data collected from 1994 to 2007 on 39 Austroads long-term pavement performance 
(LTPP) sections and 40 Austroads long-term pavement performance maintenance (LTPPM) sections. 
Additional supplementary experimental data from the accelerated loading facility (ALF) was also used. A 
further 10 years of observational data was collected on the Austroads LTPP/LTPPM sites in the period from 
2008 to 2018. Furthermore, the quality, coverage, and scope of network observational data has also been 
significantly extended with the introduction of the traffic speed deflectometer (TSD) in 2014. The TSD 
provides substantially more detailed condition and structural data collected concurrently across the state and 
territory arterial road networks.  
The aim of the project was to update the current Austroads RD models using the additional LTPP/LTPPM 
data and other data available, such as the TSD datasets, to improve these models’ explanatory power to 
support wider loading and climate conditions observed in Australia. A proof-of-concept approach based on 
using the available TSD datasets was undertaken early in the project to determine if these datasets could be 
used initially for developing a refined structural RD model with the possibility of this approach being applied 
to the other RD models.  
The updated cracking RD model is applicable to thin surfaced unbound granular base flexible pavements 
undergoing gradual deterioration. The thin surfaces include both sprayed seals and thin asphalt (≤ 40 mm) 
surfaces. 
The cracking RD model documented in this report was based on a mechanistic-empirical, deterministic 
approach using a multi-variate non-linear regression analysis. The cracking RD model uses cumulative 
cracking, ∑∆crx, as the dependent variable for thinly surfaced flexible unbound pavements. The model was 
developed using the New South Wales dataset collected by the TSD from 2014 to 2018. The model has two 
significant independent variables: cracking age, crxAGE, and the Thornthwaite Moisture Index, TMI, as 
shown in Equation 1. Although the independent variables are statistically significant, the model has an 
indeterminant fit (r2) to the data. Consequently, the cracking RD model will need calibration using the 
calibration coefficient, Kc, to suit locally observed cracking conditions as the evidence suggests that location 
also has a bearing on the rates of cracking progression. 
∑Δcrx = Kc × [ EXP[ 0.227 × crxAGE × ( 100 + TMI )/100 ] − 1 ] 
1 
As the cracking RD model development was highly dependent on the accuracy of the estimation of sprayed 
seal life and knowledge of the sprayed seal age, further refinement of sprayed seal life models needs to be 
undertaken as well as being able to accurately determine the types of bitumen and additives used in the 
binders being assessed for cracking by the TSD. This will allow application of the most appropriate seal life 
models, when available, to determine with a reasonable degree of reliability the sprayed seal life and hence 
the cracking age.  
An alternative probabilistic approach to estimating crack initiation based on multivariate logistic regression 
with sprayed seal samples that are uncracked and lightly cracked should also be considered. This could be 
in conjunction with the current Queensland Transport and Main Roads NACOE research program aimed at 
developing reliable sprayed seal life models.  
However, further refinement of the cracking RD models should not be undertaken until several more years of 
TSD cracking data become available across most states and territories in conjunction with reliable sprayed 
seal life models. 


## Page 4
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page i 
Contents 
Summary ......................................................................................................................................................... i 
1. 
Introduction ............................................................................................................................................ 1 
1.1 Background ...................................................................................................................................... 1 
1.2 Purpose ............................................................................................................................................ 1 
1.3 Scope ............................................................................................................................................... 2 
1.4 Methodology ..................................................................................................................................... 2 
2. 
Literature Review ................................................................................................................................... 3 
2.1 Overview........................................................................................................................................... 3 
2.2 Pavement Performance .................................................................................................................... 3 
2.2.1 
Limit of Gradual Deterioration ............................................................................................. 5 
2.3 Assessment of Pavement Performance ........................................................................................... 6 
2.3.1 
Functional Performance Measures ..................................................................................... 6 
2.3.2 
Structural Indicators............................................................................................................. 6 
2.4 Review of Recent Modelling Practice ............................................................................................... 6 
2.4.1 
Typical Modelling Approaches ............................................................................................ 6 
2.4.2 
Machine Learning with Big Datasets ................................................................................... 7 
2.4.3 
Application of ML to Pavement Deterioration Modelling ..................................................... 7 
2.5 Selection of Appropriate Modelling Approaches .............................................................................. 7 
2.5.1 
Expected Modelling Approach ............................................................................................. 8 
2.5.2 
Proof of Concept for TSD Datasets ..................................................................................... 8 
3. 
Data Review ............................................................................................................................................ 9 
3.1 Data for Analysis .............................................................................................................................. 9 
3.2 Data Requested from Road Agencies .............................................................................................. 9 
3.3 Review of the Data from Australian Road Agencies ...................................................................... 10 
3.3.1 
Transport for New South Wales (TfNSW) Data ................................................................ 10 
3.3.2 
Queensland Department of Transport and Main Roads (TMR) Data ............................... 10 
3.3.3 
Main Roads Western Australia (MRWA) Data .................................................................. 10 
3.3.4 
Department of Planning, Transport and Infrastructure (DPTI) South Australia Data ........ 11 
3.3.5 
Department of Infrastructure, Planning and Logistics (DIPL) Northern Territory Data ..... 11 
3.3.6 
Department of State Growth (DSG) Tasmania Data ......................................................... 11 
3.3.7 
Summary of Road Agency Data Supplied ......................................................................... 11 
3.4 Missing Data for RD Model Development ...................................................................................... 11 
3.4.1 
Climate Information ........................................................................................................... 11 
3.4.2 
Soil Reactivity .................................................................................................................... 12 
3.4.3 
Sub-soil Drainage Information ........................................................................................... 13 
3.4.4 
Additional Information Supplied by TfNSW ....................................................................... 14 
3.5 Review of the LTPP/LTPPM Data .................................................................................................. 14 
3.5.1 
Overview of Austroads LTPP/LTPPM Data ...................................................................... 14 
3.5.2 
Usability of Austroads LTPP/LTPPM Data ........................................................................ 17 
3.5.3 
Usability of New Zealand LTPP Data ................................................................................ 18 


## Page 5
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page ii 
4. 
Development of the Cracking RD Model ........................................................................................... 20 
4.1 General Approach and Limitations ................................................................................................. 20 
4.1.1 
Data Summary ................................................................................................................... 20 
4.2 Treatment of Independent Variables .............................................................................................. 22 
4.3 Development of the Cracking RD Model ........................................................................................ 22 
4.3.1 
Design Deflection D0 based on the D0/D0i Structural RD Model ....................................... 22 
4.3.2 
Estimation of Cracking Age, crxAGE, of the TSD Cracking Data ..................................... 22 
4.4 Cumulative Cracking, ∑∆crx, RD Model Development .................................................................. 24 
4.4.1 
The Cracking RD Model .................................................................................................... 24 
4.4.2 
Annual Incremental Cracking ............................................................................................ 25 
4.4.3 
Comparison with Austroads 2010 Cracking RD Model ..................................................... 25 
4.4.4 
Potential for Further Refinement of Cumulative Cracking Prediction ................................ 26 
5. 
Conclusions .......................................................................................................................................... 28 
References ................................................................................................................................................... 29 
Appendix A 
Multivariate Non-Linear Regression Analysis Outputs ............................................... 32 
 
 
Tables 
Table 3.1: 
Austroads LTPP and LTPPM sites .............................................................................................. 15 
Table 3.2: 
Condition progression rates for Austroads LTPP sites ................................................................ 15 
Table 4.1: 
Summary of observational data used in the cracking RD model development ........................... 21 
 
Figures 
Figure 2.1: Three stages of pavement performance ........................................................................................ 4 
Figure 2.2: Three phases of road deterioration (RD) ....................................................................................... 4 
Figure 2.3: Mean rut (mm) vs roughness (IRI) for ALF maintenance experiments .......................................... 5 
Figure 3.1: Requested data types .................................................................................................................. 10 
Figure 3.2: Distribution of TMI for TfNSW road network (2018 values) ......................................................... 12 
Figure 3.3: Soil classification based on reactivity for TFNSW road network (2018 values) ........................... 13 
Figure 3.4: Lower soil moisture (%) for TfNSW road network (2018 values) ................................................. 13 
Figure 3.5: Austroads LTPP and LTPPM sites .............................................................................................. 15 
Figure 3.6: Distribution of various parameters − Austroads LTPP sites ........................................................ 16 
Figure 3.7: Distribution of various parameters − Austroads LTPPM sites ..................................................... 17 
Figure 3.8: Distribution of various parameters − NZTA LTPP sites ............................................................... 19 
Figure 4.1: Prediction of cumulative cracking................................................................................................. 25 
Figure 4.2: ∑∆crx vs crxAGE .......................................................................................................................... 26 
 
 


## Page 6
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 1 
1. Introduction 
1.1 
Background 
Road agencies and industry are increasingly required to predict pavement condition and make justifiable 
decisions using information and analyses from their pavement management systems (PMS) to: 
• support future funding needs and to identify preservation and renewal candidates, accounting for the 
requirements of different roads and regions including road user requirements  
• justify possible charges to road users, including those for heavy vehicles as part of heavy vehicle road 
reform (HVRR). 
In each case a defendable basis is needed requiring an evidence-based understanding of how roads 
deteriorate over time considering traffic and environmental loading as well as maintenance practices. The full 
range of sealed road circumstances must also be covered, including different design standards and asset 
provision and condition (including surface and pavement types, shoulders, drains, etc.) with a need to 
incorporate significant variables in the predictive models based on their intended application. Terrain and 
climatic conditions are also important, with topography and subgrade soils impacting performance, as well as 
consideration of the contribution of long term and short term, often severe climatic patterns. 
Road deterioration (RD) models help in predicting future road condition and assessing maintenance needs. 
The current Austroads RD models available for use by road agencies and industry were developed in 2008 
(Austroads 2010a, 2010b) using observational data collected from 1994 to 2007 on 39 long-term pavement 
performance (LTPP) sections and 40 long-term pavement performance maintenance (LTPPM) sections, 
including additional supplementary experimental data (Martin 2010a, 2010b) from the accelerated loading 
facility (ALF).  
A New Zealand LTPP program was established in 2001 with 176 sites and now has over 18 years of 
observational data. This program was subject to a review by Henning and Brown (2016) recommending that 
further information be provided to refine the current RD models used in New Zealand. 
1.2 
Purpose 
The purpose of the project was to update the four current Austroads RD models using the additional data 
and information available to improve their explanatory power to support wider loading and climate conditions 
observed in Australia. A further 10 years of observational data was collected on the Austroads LTPP/LTPPM 
sites in the period from 2008 to 2018.  
Furthermore, the quality, coverage and scope of network observational data has also improved significantly 
with the introduction of the traffic speed deflectometer (TSD) in 2014. The TSD provides substantially more 
detailed surface and structural condition data collected concurrently across the state arterial road networks. 
By incorporating this wealth of information in model development, including the TSD data, the acceptance 
and usefulness of the resulting more robust models for road agencies and industry was expected to be much 
greater. 


## Page 7
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 2 
1.3 
Scope 
The four RD models updated represent flexible pavements, i.e. sprayed seal and thin asphalt on unbound 
granular pavements. These RD models are: 
1. cracking deterioration model 
2. rutting deterioration model 
3. roughness deterioration model 
4. structural (deflection) deterioration model. 
1.4 
Methodology 
The approach to updating the RD models included a literature review to gain state-of-art approaches to RD 
modelling, presentation of modelling concepts, workshops, and consultation with the project working group 
throughout the project duration. 
The methodology adopted evolved as the project progressed based on the availability, quality and duration 
of the data collected. Section 4 details the methodology adopted for the cracking RD model.  
 
 


## Page 8
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 3 
2. Literature Review 
2.1 
Overview 
A literature review was undertaken to gain a state-of-the-art appreciation of current practice in RD model 
development and refinement to ensure that the updating of the current Austroads RD models was 
undertaken using best practice. These RD models were based on a mechanistic-empirical deterministic 
approach using the observational and experimental data available in 2008.  
The review included an analysis of the potential for feasible RD model development using the vast amount of 
data gained from the TSD measurement of road conditions, regarded as ‘big data’, over several years in 
New South Wales and Queensland, and more recently in Western Australia. The TSD data was collected 
every year for five to six years involving 17 000–20 000 km of road length in New South Wales and 
Queensland, respectively. Western Australia and New Zealand have also had TSD network level surveys. In 
2013 the National Research Council (2013) considered model building with big data to be relatively 
uncharted territory.  
The review included a proof-of-concept investigation of the machine learning, or artificial intelligence (AI), 
approach made possible with the TSD data. The expectation was that ‘big data’ analytics applied to the TSD 
datasets may prove useful in identifying contributing factors to pavement deterioration and condition and 
guide the selection of the independent variables to be included in the RD model update. However, it was 
considered that it was not advisable to combine the TSD data with the relatively small LTPP/LTPPM dataset.  
2.2 
Pavement Performance 
There are two main stages of in-service pavement performance for which most RD models apply (Martin & 
Choummanivong 2018). These are: 
• 
RD of pavement condition, both functional and structural, immediately post construction. See Stage (1) 
in Figure 2.1  
• 
RD of pavement conditions, both functional and structural, post works effects. See Stage (3) in 
Figure 2.1.  
Stage (2) in Figure 2.1 is the impact of works effects (WE) which restore pavement conditions closer to what 
they were after original construction, although the actual effect will differ depending on treatment applied. It is 
usually assumed in practice that the same RD models apply to Stages (1) and (3), although more 
sophisticated modelling may be able to account for changes in structural conditions due to the WE which can 
influence the predicted rate of deterioration. 


## Page 9
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 4 
Figure 2.1: Three stages of pavement performance 
 
Source: Martin and Choummanivong (2018). 
Figure 2.2 shows the three phases of road deterioration, or distress, with the passage of time and traffic load. 
The first phase, initial densification, occurs within the first 12 months post construction with traffic loading. 
Initial densification is apparent in certain forms of permanent distress such as rutting and roughness. Initial 
densification may also lead to an increase in pavement strength, but this is not usually accounted for in 
deterministic RD modelling. 
Figure 2.2: Three phases of road deterioration (RD) 
 
Source: Martin and Choummanivong (2018).  
The second phase of road deterioration, gradual deterioration, is the phase in which most in-service 
pavements operate. As Figure 2.2 shows, gradual deterioration proceeds at a low rate and is virtually linear. 
Most of the documented Austroads RD models (Austroads 2010a, 2010b) are confined to predictions in the 
gradual deterioration phase.  
The rapid deterioration phase is the third and final phase of deterioration. The rapid deterioration phase, 
defined as a deterioration rate two to three times that of the gradual deterioration rate, is shown in Figure 2.2. 
Rapid deterioration is difficult to predict as the pavement approaches ultimate failure. Some RD models were 
developed for this phase (Austroads 2014, 2015); however, they are currently not fully reliable predictors of 
rapid deterioration due to the limited observational data.  


## Page 10
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 5 
2.2.1 Limit of Gradual Deterioration 
The three phases of deterioration were simulated by ALF testing of various surface maintenance treatments 
and formed the basis for defining the frontier limiting the gradual deterioration phase of sealed granular 
pavements. This frontier was defined using a binary logistic regression analysis (Mendenhall & Sincich 1996) 
of the total of 71 samples identified undergoing either gradual or rapid deterioration (Martin 2008). The 
samples identified as undergoing rapid deterioration were defined as those deteriorating at a rate of two to 
three times the samples undergoing gradual deterioration.  
Figure 2.3 plots the 53 samples defined as experiencing gradual deterioration as open squares and the 
18 samples defined as experiencing rapid deterioration as filled squares. The frontier between gradual and 
rapid deterioration was defined as in Equation 2. 
rutmax = 86.347 − 11.008 × IRI 
2 
where 
rutmax = 
mean maximum vertical deformation from the original surface profile (mm) with an absolute 
maximum value of 25 mm 
IRI = 
International Roughness Index (m/km) 
 
All the coefficients and the constant term in Equation 2 had relatively low values of standard error (SE) all 
with significance at the 95% confidence level. Equation 2 predicted the correct state of the gradual and rapid 
deterioration for over 94% of the total of 71 samples. 
Figure 2.3: Mean rut (mm) vs roughness (IRI) for ALF maintenance experiments 
 
Source: Martin (2008).  
0
10
20
30
40
50
0
5
10
15
20
IRI (m/km)
Mean rut (mm)
Gradual deterioration samples
Rapid deterioration samples
Frontier prediction by logistic regression 
Onset of rapid deterioration (Logit model y = 0) 
( rut = 86.347 − 11.008 × IRI )
Max. rut depth
 = 25 mm
7.84


## Page 11
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 6 
2.3 
Assessment of Pavement Performance  
2.3.1 Functional Performance Measures 
Functional performance measures such as road roughness, rutting and cracking are used as measures, or 
indicators, of long-term pavement performance. Road roughness, as a functional performance measure, has 
the distinct advantage of being objectively measured by a standardised measuring device, unlike some 
pavement performance indices. A commonly used measure of roughness, the International Roughness Index 
(IRI in m/km), is defined by a polynomial function of the root mean square deviation (RMSD) from a 
straight-line profile measured at given intervals along a given base length (Sayers, Gillespie & 
Queiroz 1986).  
2.3.2 Structural Indicators 
Some form of structural performance indicator is needed to assess this aspect of pavement performance. 
While visible surface distress is a useful indicator of surface functional condition, it does not give any direct 
indication of structural condition. For example, surface cracking is indistinguishable from full depth structural 
cracking and sub-surface cracking is not visible from the surface (Eijbersen and van Zwieten 1998).  
Structural capacity can be reduced either gradually or rapidly by the unplanned increased passage of traffic 
load (equivalent standard axles, ESAs) over time and rapidly by local shear failure of the pavement from 
increased axle loads. Road agencies have also increasingly decided to include the structural condition of 
pavements as a performance indicator for reporting asset condition, and in specific cases for monitoring 
contractor managed pavements on behalf of the road agency.   
2.4 
Review of Recent Modelling Practice  
As the current project is aimed at updating the existing Austroads road deterioration models developed in 
2008, the literature review covered work done on the development, refinement, calibration, and validation of 
flexible pavement deterioration models post 2006. 
2.4.1 Typical Modelling Approaches 
The following approaches are normally used to predict pavement performance (Haas, Hudson & 
Zaniewski 1994, Lytton 1987, Mahoney 1990): 
• Probabilistic, or trend, approaches that inherently recognise the stochastic nature of pavement 
performance by predicting the variability of the dependent variable for both structural and functional 
distress deterioration  
– probabilistic approaches include Markov chain models (Haas, Hudson & Zaniewski 1994), 
semi-Markov chain models (Lytton 1987), Bayesian models (National Research Council 2013), logistic 
regression models (Austroads 2012) and fuzzy logic models (Jooste et al. 2010).  
• Deterministic approaches that predict a single value of the dependent variable from pavement 
performance prediction models are based on statistical relationships between the dependent and 
independent pavement performance indicators  
– deterministic approaches include mechanistic models (Rauhut & Gendell 1987), mechanistic-empirical 
and empirical models (Lytton 1987), and the artificial neural networks (ANN) approach (Abdelaziz et 
al. 2020).  


## Page 12
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 7 
2.4.2 Machine Learning with Big Datasets 
Machine learning techniques typically require very large datasets to train models or algorithms that can be 
used for prediction. These algorithms can provide robust models although they can also be prone to 
overfitting where a model will fit the training data too closely and not be transferrable to other data. The other 
challenge with machine learning models is interpretability of the models. While machine learning black-box 
models are frequently successful in producing robust models, there can be a mistrust of the results (Gilpin, 
Bau, Yuan, Bajwa, Specter & Kagal 2019) or they require a complex implementation.  
Clustering 
Clustering is an activity that partitions big data into groups so that data items within each group are similar to 
each other and items across different groups are not similar. K-means and hierarchical clustering are popular 
algorithmic approaches. Mixture models use a probabilistic framework to model clusters (National Research 
Council 2013).  
Big data is highly amenable to the data analysis algorithms used in machine learning and statistical learning. 
A k-means algorithm is a useful method for the clustering of data into groups that is not based on a statistical 
model of the data. Optimisation formulations are needed to find the inferences from the data and determine 
whether the data can be clustered into coherent groups.  
Machine learning (ML) models 
Three main classes of machine learning models are presented in Simeone (2018), which are:  
1. supervised learning, where the training data has a known target or output variable  
2. unsupervised learning, where there is no output variable, and the data analysis aims to find patterns and 
structures in the data 
3. reinforcement learning where the model receives feedback on decisions rather than having a clearly 
known output. 
2.4.3 Application of ML to Pavement Deterioration Modelling 
Tabatabae, Shafahi & Ziyadi (2013) developed a two-stage model for predicting pavement deterioration 
using data from the Minnesota Department of Transportation hot mix asphalt test pavement between 1993 
and 2008. The model first classifies the data into like groups or families (clusters) using a support vector 
machine (SVM) model (a supervised ML model with associated learning algorithms that analyse data for 
classification and regression analysis). A recurrent neural networks (RNN) artificial intelligence (AI) approach 
is then used to model the deterioration. RNN models can handle time-series data (Choi & Do 2019).  
2.5 
Selection of Appropriate Modelling Approaches 
The literature revealed the successful application of probabilistic and deterministic approaches in pavement 
performance modelling with varying degrees of complexity and limitations due to the availability of suitable 
data. Models generated through the probabilistic approach generally provide higher prediction accuracy over 
the deterministic models. However, deterministic equation-based models are relatively straightforward 
showing the interaction between predictor parameters and modelled output and are more transferable than 
probabilistic models due to their incorporation of multiple explanatory variables.  
The decision on the selection of an appropriate modelling approach is contingent upon its intended 
application. The choice of modelling methodology depends on the following:  
• availability of sufficient high-quality data 
• preference of the road agencies (e.g. equation-based model versus ANN-based model) 


## Page 13
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 8 
• knowledge and expertise on a particular domain (e.g. machine learning, deep learning, etc.) 
• the complexity of the implementation within a PMS system 
• cost of implementation.  
Currently most Austroads members use deterministic RD models. The reasons for this choice include:  
• In-service pavement deterioration occurs in the gradual deterioration phase, which is where deterministic 
model predictions are likely to be most stable.  
• RD models can be easily incorporated into a PMS.  
• RD models can provide a relationship to traffic data which is important to consider with rising traffic 
volumes.  
• The outputs of these models can be calibrated to reflect observed pavement performance under various 
loading and environmental conditions, and service level requirements. 
2.5.1 Expected Modelling Approach 
In this context of updating the existing RD models using the high-quality LTPP/LTPPM dataset it was 
expected that the current mechanistic-empirical deterministic approach will be used.  
Kadar et al. (2015) have shown, on a proof-of-concept basis, that a deterministic RD model in combination 
with an observationally based probability density function (PDF) can be used to give both single value and 
probabilistic estimates of pavement conditions over time.  
2.5.2 Proof of Concept for TSD Datasets 
A proof-of-concept approach based on using the available TSD datasets was undertaken to determine if this 
approach could be used initially for developing a structural RD model with the possibility of this approach 
being used for the other RD models.  
The independent variables currently not included in the existing structural RD model such as surface 
distresses, subgrade reactivity, terrain category and drainage conditions were considered for inclusion in the 
model in the proof-of-concept approach. The availability of time series TSD data along with the required 
explanatory variables (climate, drainage, soil classification etc.) determined the selection of network data that 
was used for the development of a proof-of-concept structural RD model.  


## Page 14
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 9 
3. Data Review 
3.1 
Data for Analysis 
A review of the supplied historical condition and data received from road agencies and the LTPP/LTPPM 
datasets from Australia and New Zealand was undertaken. Data was reviewed to check the completeness of 
the supplied datasets with the inclusion and availability of the following additional information that potentially 
provided additional explanatory independent variables for updating the current RD models:  
• time series TSD data 
• pavement inventory information including terrain and drainage conditions 
• climate information 
• maintenance information.  
The project also used a TSD dataset from Transport for NSW (TfNSW) in a proof-of-concept approach to 
determine its usefulness in developing a refined structural RD model taking into consideration the 
independent variables currently not included such as: surface distresses, subgrade reactivity, terrain 
category and drainage conditions. The structural RD model was seen as a critical element in the RD model 
update as it is an input variable for both the rutting and roughness RD models. The structural RD model may 
also be an input variable for the RD models for cracking in some cases. The Austroads LTPP/LTPPM 
datasets were expected to be primarily used to update the current RD models due to their completeness, as 
noted in Section 3.5.2. However, the cracking data in the TSD dataset from TfNSW was used to develop the 
cracking RD model.   
The LTPP/LTPPM and the New Zealand Transport Agency (NZTA) LTPP datasets contained most of the 
additional independent variables being considered for the RD model upgrade. The state and territory road 
agencies did not have network level deflection data, except NSW and Queensland with TSD datasets. The 
LTPP/LTPPM and NZTA LTPP datasets were not combined with the TSD deflection datasets because the 
‘big data’ TSD datasets would overwhelm the LTPP datasets, as further discussed in Section 3.5.3. The 
other problem with combining the deflection datasets of different sizes is the difference in the length of the 
time series of the condition data: the NSW TSD dataset has a time series of five years while the 
LTPP/LTPPM datasets had a time series ranging from 18 to 24 years.  
3.2 
Data Requested from Road Agencies 
A data template prepared by the Australian Road Research Board (ARRB) was also distributed to the road 
agencies for supply of the required information. Figure 3.1 details the list of the requested data types which 
included the following: 
• section level information 
• inventory information 
• condition data 
• traffic information 
• climate/environmental information 
• maintenance history. 


## Page 15
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 10 
Figure 3.1: Requested data types 
 
3.3 
Review of the Data from Australian Road Agencies 
The supplied dataset from each road agency was reviewed separately to assess the completeness of the 
dataset as well as its coverage over time. The following sections summarise the findings from each road 
agency dataset.  
3.3.1 Transport for New South Wales (TfNSW) Data 
For the data supplied by TfNSW, surface condition information was available from 1999 onwards. The TSD 
data was collected every year for five years involving 15 000–20 000 km of road length in New South Wales. 
Consequently, deflection data was available from 2014 to 2018. This TSD data included cracking data which 
was subsequently used for the cracking RD model development as discussed in Section 4.1.  
The TSD data collected was 1 800 km road length with time series data for three years, 9 000 km with time 
series data for four years and 4 000 km with time series data for five years covering 11% of the arterial road 
network. 
3.3.2 Queensland Department of Transport and Main Roads (TMR) Data 
For the data supplied by TMR, surface condition information was available from 2015 to 2020. The TSD data 
was collected every year for five years involving approximately 15 000 km of road length in Queensland. 
Therefore, deflection data was available from 2015 to 2020.  
The TSD data collected around 5 000 km road length with time series data for three years, 3 400 km with 
time series data for four years and 7 800 km with time series data for five years covering 22% of the arterial 
road network. 
3.3.3 Main Roads Western Australia (MRWA) Data 
Surface condition data was supplied by MRWA from 2009 onwards. However, TSD deflection and cracking 
data was only available for 2019 (2019 data was collected over 2018 to 2019). Falling weight deflectometer 
(FWD) data was also supplied with a partial collection from 2002 to 2017.  


## Page 16
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 11 
MRWA 2019 TSD data was collected for the full network (17 000 km) over 2018 to 2019.  
3.3.4 Department of Planning, Transport and Infrastructure (DPTI) South Australia 
Data 
DPTI supplied surface condition data (roughness and rutting) from 2009 to 2020. Cracking data was only 
available over the period from 2017 to 2020 with two years of data collected. The DPTI did not have any TSD 
data collected, and no FWD deflection data was supplied. 
3.3.5 Department of Infrastructure, Planning and Logistics (DIPL) Northern 
Territory Data 
DIPL supplied surface condition data (roughness, rutting, cracking) for two years (2015, 2019). DIPL also 
had TSD deflection data collected for the whole network in 2019 which was supplied.  
3.3.6 Department of State Growth (DSG) Tasmania Data 
Surface condition data (roughness, rutting and cracking) supplied by DSG included for three years (2012, 
2015, 2018) of data. TSD deflection data was also collected for part of the network in 2018. 
3.3.7 Summary of Road Agency Data Supplied 
Findings from the review of the road agency data supplied were as follows: 
• Most road agencies had time series surface condition information (roughness, rutting and cracking). 
• A lack of complete network-wide time series deflection data required for the structural model except for 
TfNSW and TMR. 
• A lack of important ancillary information e.g. maintenance history, drainage, climate, etc. 
3.4 
Missing Data for RD Model Development  
The following missing independent variables were estimated initially for the proof-of-concept approach and 
then later used as additional explanatory independent variables for refinement of the roughness, rutting and 
cracking RD models.  
3.4.1 Climate Information 
Historical climate information (monthly minimum, maximum and average temperature, and monthly rainfall 
information for years 1994 to 2018) was supplied from the Bureau of Meteorology website (BOM 2021) for: 
• the weather stations close to the LTPP/LTPPM  
• for 30 climate districts in NSW  
• information on the climate for each road link.  
The supplied data was used to calculate Thornthwaite Moisture Index (TMI) (Thornthwaite 1948) for years 
2014 to 2018 for the whole network using TMI calculation method 3 from a previous project 
(Austroads 2010c). 
Figure 3.2 shows the distribution of TMI over the TfNSW road network (for year 2018) and its comparison 
against the TMI map for the whole of Australia. The calculated TMI values are drier than those observed in 
Austroads (2004), but the relative distribution is similar to those on the Australian TMI map. 


## Page 17
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 12 
Figure 3.2: Distribution of TMI for TfNSW road network (2018 values) 
 
3.4.2 Soil Reactivity 
The degree of soil reactivity determines the swelling and shrinkage behaviour of the soil with changes in 
moisture content. While sandy soils are generally non-reactive, a few types of clayey soils are highly reactive 
due to their smectite clay mineralogy. No direct information, in terms of the amount of subgrade soil 
reactivity, was available for any of the road networks. The data available from different public sources 
(CSIRO 2013, Viscarra Rossel 2014) was used to estimate the reactivity information based on clay content 
(%Clay, 0−30 mm) and clay mineralogy information. This was done as follows:  
• The clay content map was compared with the clay mineralogy map (Viscarra Rossel 2014). It was found 
that the clay mineralogy of the subsoil lined up with the clay content with higher clay contents associated 
with the reactive clay minerals (montmorillonite/smectites). 
• QGIS is a geographical information system (GIS) that has software allowing users to analyse and edit 
spatial information. In this application QGIS uses a point sampling technique to estimate clay content 
information on the TfNSW road segments. 
• Assigning soil reactivity is based on the % clay content: < 30% is non-reactive; 30−45% is moderately 
reactive; and > 45% is reactive. 
Figure 3.3 displays the estimated clay reactivity for the TfNSW network for 2018. Since the surveyed network 
varied from year to year, soil reactivity information was estimated for all segments surveyed in each year. 


## Page 18
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 13 
Figure 3.3: Soil classification based on reactivity for TFNSW road network (2018 values) 
 
Sources: CSIRO (2013), Viscarra Rossel (2014). 
3.4.3 Sub-soil Drainage Information 
TfNSW was the only road agency able to provide terrain information in their initial data supply although no 
drainage condition data was supplied. In the absence of drainage information in NSW, the availability of the 
lower soil moisture (LSM) content percentage (%) was extracted for use as a surrogate to estimate drainage 
conditions. The historical LSM (2001−2021) percentage was extracted from the Bureau of Meteorology 
website (BOM 2021) for all of Australia.  
Figure 3.4 displays the LSM percentage for the TfNSW network for 2018. Since the surveyed network varied 
from year to year, the LSM percentage information was estimated for all segments surveyed in each year. 
Figure 3.4: Lower soil moisture (%) for TfNSW road network (2018 values) 
 
Source: BOM (2021). 


## Page 19
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 14 
3.4.4 Additional Information Supplied by TfNSW 
TfNSW provided the following additional information for the whole NSW network:  
• Maintenance history – historical routine maintenance cost ($/lane-km/year) from 2014 to 2021 for 
segments where some type of routine maintenance work was done.  
• Asphalt thickness – estimated for the asphalt surfaced segments. The data specifies whether the 
thickness is more or less than 100 mm. 
• Seal type information – provided for the sprayed seal surfaced segments. The data specified whether the 
seal type was single/single, double/double etc.  
3.5 
Review of the LTPP/LTPPM Data 
LTPP data from both Australia and New Zealand was reviewed by the project. After the development of 
current Austroads RD models in 2010, a further 10 years of observational data was collected on the 
Austroads LTPP/LTPPM sites in the period from 2008 to 2018. The New Zealand NZTA LTPP program was 
established in 2001 with 176 sites providing 18 years of observational data. The proposed methodology 
primarily aimed to use the LTPP/LTPPM and NZTA LTPP datasets to update the current RD models.  
3.5.1 Overview of Austroads LTPP/LTPPM Data  
The Austroads LTPP/LTPPM program covered the following:  
• 33 LTPP sites (39 sections) 
• 8 LTPPM sites (40 sections) 
• sites across all Australian jurisdictions except Western Australia and Northern Territory. 
The Austroads LTPP/LTPPM database contained a detailed record of the following:  
• section details including pavement composition 
• roughness, rut depth, falling weight deflectometer (FWD) data 
• maintenance history 
• traffic 
• environment, TMI.  
Figure 3.5 shows the location of the LTPP and LTPPM sites across Australia, while Table 3.1 
outlines features of the site groups and the number of sites and Table 3.2 summarises the observed 
performance of the site groups. 


## Page 20
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 15 
Figure 3.5: Austroads LTPP and LTPPM sites 
 
Source: Austroads (2019b). 
Table 3.1: Austroads LTPP and LTPPM sites  
All pavement groups 
No of sites 
No of sections 
Section length 
(m) 
Test regime 
LTPP 
23 
23 
100–200 
SHRP − LTPP 
ACT-LTPP 
6 
12 
800 
Network survey 
SA-LTPP 
4 
4 
12 
ARRB method 
LTPPM 
8 
40 
200 
Reduced SHRP − 
LTPP 
Source: Austroads (2019b). 
Table 3.2: Condition progression rates for Austroads LTPP sites  
Pavement group 
Deflection 
(micron/yr) 
Roughness 
(IRI/yr) 
Rutting 
(mm/yr) 
Cracking 
(%/yr) 
Traffic 
(MESA/yr) 
LTPP (SHRP+ALF) 
11 
0.07 
0.20 
1.92 
0.4−8.8 
LTPP (AAPA) 
7 
0.04 
0.26 
0.06 
3.0–7.5 
LTPP (rural highway) 
72 
0.03 
0.78 
0.06 
0.02–0.8 
ACT-LTPP 
24 
0.05 
0.17 
0.87 
0.05–1.0 
Source: Austroads (2019b). 
Figure 3.6 and Figure 3.7 show the distribution of inner and outer wheel path roughness (IRI, m/km) and 
rutting (mm), maximum deflection (micron), rainfall (mm) and traffic (AADT) for the Austroads LTPP and 
LTPPM sites, respectively.  


## Page 21
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 16 
Figure 3.6: Distribution of various parameters − Austroads LTPP sites 
 
The shapes of separate distributions of the parameters for the LTPP and LTPPM sites are generally similar, 
except for the traffic and rainfall distributions which are much narrower for the LTPPM sites, reflecting the 
relatively small number of 8 LTPPM locations compared to the 41 LTPP locations.  
On the other hand, the maximum deflection distribution for the LTPP sites is narrower than that for the 
LTPPM sites, with a mean maximum deflection (D0 = 250 micron) and modified structural number 
(SNC = 7.7). The LTPP sites have a higher strength than the LTPPM sites that have a higher mean 
maximum deflection and lower strength (D0 = 500 micron; SNC = 5.0). The rutting and roughness 
distributions for the LTPPM sites have a greater range than the LTPP rutting and roughness distributions 
which also reflects the lower deflections (higher pavement strength) of the LTPP sites.  


## Page 22
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 17 
Figure 3.7: Distribution of various parameters − Austroads LTPPM sites 
 
3.5.2 Usability of Austroads LTPP/LTPPM Data 
The level of completeness in the Austroads LTPP/LTPPM datasets, including the availability of necessary 
additional information, such as traffic, environment, climate, maintenance history etc., make this an excellent 
basis for RD model update. However, the following data was not included in the LTPP/LTPPM datasets and 
required estimation as part of this data review and update process: 
• soil reactivity 
• drainage and terrain. 
Soil reactivity 
Soil reactivity for the Austroads LTPP/LTPPM sites was estimated using the same methods applied in 
Section 3.4.2 in estimating the soil reactivity for the TfNSW network. 
Drainage and terrain 
No terrain or drainage condition data was available for Austroads LTPP/LTPPM sites. In the absence of 
drainage information, the lower soil moisture content (LSM) data was extracted and used as a surrogate for 
drainage. This information was estimated using the same method applied in Section 3.4.3.  
It was not possible to source terrain information for the LTPP/LTPPM sections, although most sections were 
located in relatively flat terrain.  


## Page 23
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 18 
3.5.3 Usability of New Zealand LTPP Data 
Some lack of completeness in the New Zealand LTPP dataset, including the availability of necessary 
additional information e.g. climate measured using TMI and drainage measured using LSM, made this 
dataset less suitable for the RD model update.  
The Austroads and New Zealand LTPP datasets were compared using descriptive statistics and ‘t’ tests in 
IBM SPSS Statistics software 2021. The results showed that: 
• Roughness statistics (mean, median, range, etc.) were comparable between the Austroads and New 
Zealand LTPP sites. 
• Rutting statistics for New Zealand LTPP sites were slightly higher than those observed on the Austroads 
LTPP sites. 
• Deflection values were significantly different between the two datasets. The New Zealand LTPP sites 
displayed much higher mean deflections (lower strength) than the Austroads LTPP sites. 
Figure 3.8 shows the distribution of inner and outer wheel path rutting (mm), lane roughness (IRI, m/km), 
adjusted modified structural number (SNP), and rainfall (mm) for the NZTA LTPP sites. The rainfall 
distribution is significantly greater than that for the Austroads LTPP/LTPPM sites, reflecting the larger 
number of NZTA LTPP sites (167). The NZTA LTPP sites approximate mean pavement strength, SNP, is 
around 3.5, significantly lower than the range of mean pavement strength of SNC from 5 to 7.7 for the 
Austroads LTPP/LTPPM sites.  
The strength parameters, SNC and SNP are similar measures where SNP uses a weighting factor (Parkman 
& Rolt 1997) to SNC that reduces the contribution to pavement strength of the subbase and subgrade with 
their increasing depth. This weighting factor was introduced because the strength contribution of each layer 
in the pavement system is independent of depth with the SNC parameter resulting in over-estimates of SNC 
in the case of pavements greater than 700 mm thick. 


## Page 24
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 19 
Figure 3.8: Distribution of various parameters − NZTA LTPP sites 
 
Due to the difficulty of estimating TMI and LSM variables in New Zealand and the fact that the New Zealand 
LTPP sites displayed much higher mean deflections than the Austroads LTPP sites, indicating substantial 
differences in the New Zealand subgrades compared to those in Australia, it was decided not to combine the 
Austroads and New Zealand LTPP datasets. It is also understood that New Zealand has highly resilient 
volcanic-derived soils which are atypical in comparison to Australian subgrades. 


## Page 25
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 20 
4. Development of the Cracking RD Model 
4.1 
General Approach and Limitations 
As noted in Section 3.3.1, the cracking RD model was developed using the TSD cracking data provided by 
TfNSW. The cracking RD model was limited to thinly surfaced (sprayed seal and thin asphalt ≤ 40 mm) 
flexible unbound granular base pavements undergoing gradual deterioration. This was due to:  
1. the general lack of cumulative cracking, ∑∆crx, observed on the wider ranging LTPP/LTPPM sites  
2. cracking not being accurately and routinely measured until late in the LTPP/LTPPM monitoring program 
when the automatic crack detection (ACD) device was introduced 
3. the majority of TfNSW TSD cracking data was for sprayed seal unbound granular pavements undergoing 
gradual deterioration. 
The existing Austroads cracking model was based on cracking data from South Australia (Austroads 2010a). 
The historical annual cracking rates of progression in South Australia were observed to range from 0.16 to 
4.15% cracking per year (Austroads 2008), while the LTPP/LTPPM dataset showed a range of cracking 
progression from 0.06 to 1.92% cracking per year (Austroads 2019b). The maximum level of cracking 
observed in South Australia (Austroads 2010a) was 53% while the maximum observed cracking from the 
TfNSW TSD cracking data was 18% (see Table 4.1) implying that all the TfNSW cracking data was 
undergoing gradual deterioration.  
4.1.1 Data Summary 
The complete Austroads LTPP/LTPPM database, including all sites with all years of collected data, was used 
at the beginning of the analysis to determine suitable data samples. However, not all sites could be used due 
to their pavement composition, condition progression, maintenance history etc. The following criteria were 
used to determine the suitability of the sites for analysis: 
• Sites with bound/stabilised pavement bases were excluded due to different pavement deterioration 
characteristics compared with unbound granular pavements. 
• Sites with thick asphalt surfaces (> 40 mm) were excluded. 
• For the remaining LTPP/LTPPM sites, the condition data years that were free from the effect of the 
maintenance were retained. 
• Further filtering of the above sites was done to only include years with an increasing trend in condition 
progression to exclude effects from unrecorded maintenance if any. 
The above process significantly reduced the final sample size for analysis. The following tabulation 
summarises the characteristics of the TfNSW TSD cracking data, and the independent variables considered 
in the cracking RD model development.


## Page 26
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 21 
Table 4.1: Summary of observational data used in the cracking RD model development 
Data source 
Independent variable range 
Dependent variable 
range 
No. samples 
Variables 
AGE(1) 
MESA(2) 
TMI(3) 
Clay(4) 
SL(5) 
D0i(6) 
LSMF(7) 
D0(8) 
∑∆crxi(9) 
 
TfNSW 
15−93 
0.02−6.54 
–48−12 
0−18 
– 
14.5−1250 
0.042−0.96 
5.8−636 
0−18 
257 
1. Pavement age (years) at time of observation. 
2. Traffic load in millions of equivalent standard axles per lane per year. 
3. Thornthwaite Moisture Index. 
4. Clay (%) in subgrade. 
5. Service life (years) increased SL related to increased cumulative ESAs.  
6. Measured maximum deflection (micron). 
7. Lower soil moisture. 
8. Design maximum deflection (micron).  
9. Cumulative cracking (%).  


## Page 27
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 22 
4.2 
Treatment of Independent Variables  
Section 3.4 shows how the independent variables TMI, Clay and LSM were derived for the TfNSW TSD 
dataset.  
Two potential independent variables, MESA and AGE, were supplied by TfNSW for the cumulative cracking, 
∑∆crx, model. Another possible independent variable, service life, SL, for the cumulative cracking, ∑∆crx, 
model was estimated based on the supplied value for MESA based on the approach used in previous 
Austroads (2010a) RD modelling. The SL estimates ranged from 40 to 60 years, increasing with the traffic 
capacity of the road. Estimation of the design deflection, D0, as a further possible independent structural 
variable is discussed in the next section.  
4.3 
Development of the Cracking RD Model  
4.3.1 Design Deflection D0 based on the D0/D0i Structural RD Model  
The design deflection D0 variable, or in-service initial deflection, D0 (based on Benkelman Beam deflection), 
was considered as a possible independent variable for the cumulative cracking, ∑∆crx, model based on the 
D0 dependent variable in the currently updated structural RD model. The D0 variable was derived from 
measured D0i values as in Equation 3 which were seasonally adjusted by the Austroads (2019a) approach.  
D0 = Ks × D0i × [ 1 – AGE × ( 0.012 + 0.001 × MESA + 0.0000868 × TMI ) ] 
3 
where 
Ks = 
structural calibration coefficient for adjustment to local conditions (default = 1.0) 
AGE = 
pavement age (years) 
TMI = 
Thornthwaite Moisture Index (1948) range –50 to 100) 
MESA = 
annual millions of ESAs of traffic load per lane 
4.3.2 Estimation of Cracking Age, crxAGE, of the TSD Cracking Data  
The cracking age, crxAGE, is the number of years since cracking was initiated at the end of sprayed seal life. 
The cracking age, crxAGE, is estimated based on the age of the seal and the estimated time to initiate 
surface cracking as shown in Equation 4.  
crxAGE = seal age – seal life 
4 


## Page 28
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 23 
The implication is that seal life coincides with, or is used as a surrogate for, the timing of crack initiation. 
Equation 5 estimates the seal life (Austroads 2010d) based on the time for the bitumen binder to harden, and 
therefore can be used for this purpose provided it is calibrated to local conditions. 
Seal life = 
 
0.158 × TMIN – 0.107 × R + 0.84 
 
2 
5 
 
 
0.0498 × T – 0.0216 × D – 0.000381 × S2 
 
 
 
where 
 
TMIN 
= yearly mean of the daily minimum air temperature (°C) 
T 
= (TMAX + TMIN)/2 
TMAX 
= yearly mean of the daily maximum air temperature (°C) 
D 
= the ARRB Durability Test result (days, usually taken as 8 to 10) 
S 
= nominal size of seal (nominal stone size, mm) 
R 
= risk factor with a scale from 1 (very low risk) to 10 (very high risk) 
The variables, TMIN and TMAX, were available from the Bureau of Meteorology (BOM 2021) from 
meteorology stations near the sites of the data samples. The seal age of the data samples was included in 
the TSD cracking data supplied by TfNSW. Equation 5 assumes that all cracking was initiated by binder 
hardening. However, this is not always the case, as binders have additives that can change their 
performance such as polymer modified binders (PMBs).  
Several other possible seal life estimates were also considered as the cracking age, crxAGE, is a critical 
variable in predicting cumulative cracking. This uncertainty was a significant source of error in the cracking 
RD model development. Another potential source of error was the reliability of the estimates of seal age 
which can suffer from a lack of accurate resealing records. 
The risk factor, R, in Equation 6 is also somewhat arbitrary, however, Main Roads Western Australia 
(MRWA) has quantified R in terms of both rainfall and traffic as follows (MRWA 2019) which was considered 
for use in Equation 5. 
R = 0.00097 × Rain + 0.0064 × (AADT)0.5 + 0.00001 × Rain × (AADT)0.5 
6 
where 
Rain  = 
annual rainfall (mm) 
 
AADT 
= 
average annual daily traffic (vehicles/day) 
 
In keeping with the original derivation of the seal life model (Equation 5) where seal life reflects an agency 
view of end of life, it was found that many samples already had cracking present. Therefore, applying 
Equation 5 directly was not considered useful as a predictor of crack initiation. It was then decided based on 
the experience of the team that where the surface age had reached 10 years this would be the age of crack 
initiation, or seal life. Only the samples which had a surface age greater than 10 years and were cracked and 
demonstrated an increasing trend of cumulative cracking over time were included in the cracking RD model 
dataset. However, it is recognised that the assumption applied may not reflect crack initiation timing 
throughout Australia and calibration of any model is essential. 


## Page 29
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 24 
4.4 
Cumulative Cracking, ∑∆crx, RD Model Development 
4.4.1 The Cracking RD Model 
A total of 256 cumulative cracking samples, ∑Δcrx, that were experiencing increasing cracking over the 
4 years of TSD measurement were available for predicting sprayed sealed granular pavement cracking. The 
following model (Equation 7) was developed from this data by multi-variate non-linear regression (MVNLR) 
analyses by IBM SPSS Statistics software 2021.  
∑Δcrx = Kc × [ EXP[ 0.227 × crxAGE × ( 100 + TMI )/100 ] − 1 ] 
7 
where 
 
∑Δcrx = 
cumulative cracking (% total lane area) 
= 
crocodile cracking (%) + transverse cracking (%) + longitudinal cracking (%) 
crxAGE = 
cracking age (elapsed time from the commencement of cracking, years), see 
Equation 4 
TMI = 
Thornthwaite Moisture Index at time ‘i’   
Kc = 
calibration coefficient for local conditions (default = 1.0) 
The goodness of fit (r2) of Equation 5 to the data was indeterminate due to the high variation in the observed 
cumulative cracking data. As noted in Section 4.3.2, this was largely attributed to not being able to reliably 
determine the seal life on each of the samples. However, despite the lack of data fit, the crxAGE and TMI 
were the two independent variables that were statistically significant. Appendix A provides the outputs of this 
statistical analysis, although they are limited due to the lack of fit to the data and there being only two 
statistically significant variables.  
None of the other independent variables examined, MESA, D0, %Clay and lower soil moisture content factor, 
LSMF, were statistically significant. The %Clay and LSMF independent variables were correlated with TMI 
and therefore were not included. 
Figure 4.1 shows the major variation in the rate of cumulative cracking, ∑Δcrx, due to the wide range of TMI 
values with cracking age, crxAGE. 


## Page 30
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 25 
Figure 4.1: Prediction of cumulative cracking  
 
This model predicts that under a high TMI of 100, the sprayed seal will reach 100% cracking after 10 years, 
while for a lower TMI of 30, the sprayed seal will reach 100% cracking after 15 years. Under the lowest 
possible TMI of –50, the model predicts over 50% cracking after 35 years.  
4.4.2 Annual Incremental Cracking  
The annual increase in incremental cracking, Δcrx(t), is simply found by subtracting the cumulative cracking 
at time ‘i + 1’, ∆crx(t)I + i, from the cumulative cracking at time ‘i’, ∆cx(t)i, provided the cracking time, crxAGE is 
known. This approach is expressed as shown in Equation 8 using Equation 7.  
∆crx(t) = Kc × { [ EXP[ 0.227 × crxAGEI + i × ( 100 + TMI )/100 ] − 1 ] – [ EXP[ 0.227 × crxAGEi × 
( 100 + TMI )/100 ] − 1 ] } 
8 
where all terms are as previously defined. 
 
4.4.3 Comparison with Austroads 2010 Cracking RD Model  
A direct comparison between the cumulative cracking model developed here and the existing 
Austroads (2010a) cumulative cracking model for sprayed seals can be made, even though the model forms 
are different, but meet the boundary conditions. The Austroads cumulative cracking RD model is as shown in 
Equation 9 which predicts an ‘s’ shape for cracking progression. 
∑Δcrx = Kc × [ 100 – 200 × ( 1 + EXP(( 0.234 × crxAGE/(( 200 – TMI )/25))3.5 ))-1 ] 
9 
where all the variables are as previously defined.  
Figure 4.2a and b compare the predictions of cumulative cracking using the new upgraded cracking RD 
model against the existing Austroads cracking RD model.  


## Page 31
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 26 
Figure 4.2: ∑∆crx vs crxAGE 
a. 
New model 
b. 
Austroads (2010a) 
Apart from the different functional model forms for the prediction of cracking progression, the times to reach 
100% cracking for the various values of TMI are different with the Austroads (2010a) predicting much slower 
rates of crack progression for all values of TMI. 
This variation can be attributed to some extent to the location of the samples used to develop the cracking 
models. The Austroads (2010b) cracking model used cracking data from South Australia which comprised 
1675 samples where the extent of cracking was up to 53%. The new cracking model was based on the 
TfNSW TSD cracking data collected over 4,000 km of arterial road length with time series data for five years 
covering 11% of the NSW sealed arterial road network (see Section 3.3.1). In both cases the model fit to the 
data were poor to non-existent.  
The cracking RD model will need calibration, i.e. adjustment of Kc, to suit local conditions as the current 
evidence suggests that location has a bearing on the rates of cracking progression. 
Neither of the above cumulative cracking models include traffic, MESA, and pavement design deflection, D0, 
implying that thin surface cracking is not related to traffic and pavement strength. This infers that the 
pavement is designed to match the expected traffic and the cracking is therefore not due to any structural or 
pavement material deficiency.  
4.4.4 Potential for Further Refinement of Cumulative Cracking Prediction 
Large amounts of TSD cracking data will be forthcoming in the future years from most jurisdictions in 
Australia. This will provide a large dataset of continuous cracking data, based on ACD measurement, over 
several years once the non-cracked road segments are filtered out.  
As noted in Section 4.3.2, the estimation of cracking age, crxAGE or seal life, is a critical variable in 
predicting cumulative cracking and is a major source of error in the cracking RD model development if the 
cracking age is inaccurate. Further refinement of sprayed seal life models needs to be undertaken as well as 
being able to accurately determine the types of bitumen and additives used in the binders being assessed for 
cracking by the TSD. This will allow application of the most appropriate seal life models, when available, to 
determine with a reasonable degree of reliability the sprayed seal life and hence the cracking age.  
However, there is an alternative probabilistic approach to estimating crack initiation i.e. cracking age, used 
by Henning and Brown (2016) that is purely statistical based on multivariate logistic regression using seal 
samples that are uncracked and seal samples that are lightly cracked (< 10% cracked). This approach could 
use several additional variables such as stone size, traffic, rainfall, and TMI, excluding the binder properties. 


## Page 32
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 27 
However, refinement of the cracking RD models should not be undertaken until several more years of TSD 
cracking data become available across most states and territories in conjunction with reliable sprayed seal 
life models. Models are currently being investigated by Queensland Department of Transport and Main 
Road’s (TMR) National Asset Centre of Excellence (NACOE) research program (Noya & Hwayyis 2021), and 
potentially by other agencies. 


## Page 33
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 28 
5. Conclusions 
A cracking RD model using cumulative cracking, ∑∆crx, as the dependent variable for thinly surfaced flexible 
unbound pavements, was developed using the New South Wales dataset collected by the TSD from 2014 to 
2018. The model has two significant independent variables: cracking age, crxAGE, and the Thornthwaite 
Moisture Index, TMI. Although the independent variables are statistically significant, the model has an 
indeterminant fit (r2) to the data. Consequently, the cracking RD model will need calibration to suit locally 
observed cracking conditions as the evidence suggests that location also has a bearing on the rates of 
cracking progression. 
As the cracking RD model development was highly dependent on the accuracy of the estimation of sprayed 
seal life and knowledge of the sprayed seal age, further refinement of sprayed seal life models needs to be 
undertaken as well as the supply of more reliable information on the types of bitumen and additives used in 
the binders being assessed for cracking by the TSD. This will allow application of the most appropriate seal 
life models, when available, to determine with a reasonable degree of reliability the sprayed seal life and 
hence the cracking age.  
An alternative probabilistic approach to estimating crack initiation used by Henning and Brown (2016) could 
be considered based on multivariate logistic regression using seal samples that are uncracked and lightly 
cracked. Forecasting a discrete event such as crack initiation is best simulated using a probabilistic model. 
This approach could be considered in conjunction with the TMR’s NACOE research program which is aimed 
at developing reliable sprayed seal life models.  
However, refinement of the cracking RD models should not be undertaken until several more years of TSD 
cracking data become available across most states and territories in conjunction with reliable sprayed seal 
life models. 


## Page 34
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 29 
References 
Abdelaziz N, Abd-El-Hakim R T, El-Badawy S M and Hafez A (2020) ‘International roughness index 
prediction model for flexible pavements’, International Journal of Pavement Engineering, vol. 21(1), 
doi:10.1080/10298436.2018.1441414. 
Austroads (2004) Impact of climate change on road infrastructure, AP-R243-04, Austroads, Sydney, NSW. 
Austroads (2008) Development of HDM-4 road deterioration (RD) model calibrations for sealed granular and 
asphalt roads, AP-T97-08, Austroads, Sydney, NSW. 
Austroads (2010a) Interim network level functional road deterioration models, AP-T158-10, Austroads, 
Sydney, NSW. 
Austroads (2010b) Predicting structural deterioration of pavements at a network level: interim models, 
AP-T159-10, Austroads, Sydney, NSW. 
Austroads (2010c) Impact of climate change on road performance: updating climate information for Australia, 
AP-R358-10, Austroads, Sydney, NSW. 
Austroads (2010d) Asphalt and seal life prediction models based on bitumen hardening, AP-R160-10, 
Austroads, Sydney, NSW. 
Austroads (2012) The scoping and development of probabilistic road deterioration (RD) models, AP-T201-
12, Austroads, Sydney, NSW. 
Austroads (2014) Interim road deterioration cracking model during accelerated deterioration, AP-T259-14, 
Austroads, Sydney, NSW. 
Austroads (2015) Interim road deterioration models during accelerated deterioration, AP-T291-15, 
Austroads, Sydney, NSW. 
Austroads (2019a) Guide to pavement technology part 5: pavement evaluation, AGPT05-19, Austroads, 
Sydney, NSW.  
Austroads (2019b) Long-term pavement performance study: final report, AT-343-19, Austroads, Sydney, 
NSW. 
Bureau of Meteorology (2021) Australian landscape water balance: root zone soil moisture, AWRA-L model, 
BOM, Canberra, ACT, accessed 9 February 2021,  
Choi S and Do M (2019) ‘Development of the road pavement deterioration model based on the deep learning 
method, Electronics, vol. 9(3), doi:10.3390/electronics9010003 
CSIRO (2013) Australian Soil Resource Information System (ASRIS), mapping tool, CSIRO, Canberra, ACT, 
accessed 21 January 2021. 
Eijbersen M J and van Zwieten J (1998) ‘Application of FWD-measurements at the network level’ 
[conference paper], Proceedings, 4th international conference on managing pavements, vol. 1, University 
of Pretoria, South Africa.  
Gilpin L, Bau D, Yuan B, Bajwa A, Specter M and Kagal L (2019) ‘Explaining explanations: an overview of 
interpretability of machine learning’ [conference paper], IEEE 5th international conference on Data 
Science and Advanced Analytics (DSAA), 2018, pp. 80–9, doi: 10.1109/DSAA.2018.00018. 


## Page 35
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 30 
Haas R, Hudson W R and Zaniewski J (1994) Modern pavement management, Krieger Publishing Co., 
Malabar, Florida, USA.  
Henning T and Brown D (2016) Strategic review of New Zealand LTPP programme, IDS Ltd., Auckland, New 
Zealand. 
Jooste F, Esterhuysen G, Judd D and Jordaan G (2010) ‘Implementation of a rule-based deterioration model 
for dual carriageway road networks’ [conference paper], ARRB conference, 24th, 2010, Melbourne, 
Victoria, Australia, ARRB, Vermont South, Vic. 
Kadar P, Martin T, Baran M and Sen, R (2015) ‘Addressing uncertainties of performance modelling with 
stochastic information packages–incorporating uncertainty in performance and budget forecasts’ 
[conference paper], Proceedings, 9th International conference on managing pavement assets, 
Transportation Research Board, Washington, DC, USA. 
Lytton R L (1987) ‘Concepts of pavement performance and modelling’ [conference paper], Proceedings, 2nd 
North American conference on managing pavements, Ministry of Transportation, Ontario, Canada. 
Mahoney J (1990) Introduction to prediction models and performance curves, course text, FWHA Advanced 
course on pavement management, FHWA, Washington, DC, USA.  
Main Roads Western Australia (2019) Procedure pavement modelling, report no. D19#284347, MRWA, 
Perth, WA.  
Martin T (2008) ‘Predicting sealed granular pavement deterioration at a road network level’ [PhD thesis], Civil 
Engineering Department, Monash University, Clayton, Vic. 
Martin T (2010a) ‘Experimental estimation of the relative deterioration of surface maintenance treatments’, 
Journal of Transportation Engineering, vol. 136(1):1-10. 
Martin T (2010b) ‘Experimental estimation of the relative deterioration of flexible pavements under increased 
axle loads’, International Journal of Pavement Engineering, vol. 12(1):37-45.  
Martin T and Choummanivong L (2018) Predicting the performance of Australia’s arterial and sealed local 
roads, ARR 390, ARRB, Vermont South, Vic. 
Mendenhall W and Sincich T (1996) A second course in statistics: regression analysis, 5th edn, Prentice-
Hall, Inc., New Jersey, USA. 
National Research Council (2013), Frontiers in massive data analysis: chapter 7, building models from 
massive data, The National Academy Press, Washington, DC, USA.  
Noya L and Hwayyis K (2021) ‘A20 Improved model to predict the remaining life of sprayed seal surfaces 
(Year 6 – 2019/20)’, prepared for Queensland Department of Transport and Main Roads under the 
NACOE program, ARRB, Port Melbourne, Vic.  
Parkman C C and Rolt J (1997) Characterisation of pavement strength in HDM-III and possible changes for 
HDM-4, Transport Research Laboratory (TRL) unpublished report, PR/ORC/587/97, TRL, Crowthorne, 
UK. 
Rauhut J B and Gendell D S (1987) ‘Proposed development of pavement performance prediction models 
from SHRP/LTPP data' [conference paper], Proceedings, 2nd North American conference on managing 
pavements, Ministry of Transportation, Ontario, Canada. 
Sayers M W, Gillespie T D and Queiroz C A V (1986) The international road roughness experiment, technical 
paper no. 45, The World Bank, Washington DC, USA.  


## Page 36
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 31 
Simeone O (2018) ‘A very brief introduction to machine learning with application to communication systems’, 
IEEE Transactions on Cognitive Communication Systems, vol. 4(4):648–64.  
Tabatabae H, Shafahi Y and Ziyadi M (2013) ‘Two-stage support vector classifier and recurrent neural 
network predictor for pavement performance modelling’, Journal of Infrastructure Systems, vol. 19(3), 
doi:10.1061/(ASCE)IS.1943-555X.0000132. 
Thornthwaite C W (1948) ‘An approach toward rational classification of climate’, Geographical Review, 
vol. 38(1):55–94. 
Viscarra Rossel R (2014) Maps of clay minerals - kaolinite, illite and smectite in Australian soils, CSIRO Data 
Collection, CSIRO, Canberra, ACT, accessed 7 September 2022.  
 
 


## Page 37
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 32 
Appendix A Multivariate Non-Linear Regression 
Analysis Outputs 
A.1 Cumulative Cracking, ∑∆crx, RD Model  
Nonlinear Regression Analysis 
Notes 
Output created 
19-JUL-2022 12:07:40 
Comments 
Input 
Active dataset 
DataSet1 
Filter 
<none> 
Weight 
<none> 
Split file 
<none> 
No.?? of rows in working data file 
256 
Missing value handling 
Definition of missing 
User-defined missing values are 
treated as missing. 
Cases used 
Statistics are based on cases with 
no missing values for any variable 
used. Predicted values are 
calculated for cases with missing 
values on the dependent variable. 
Syntax 
MODEL PROGRAM a1=0.5. 
COMPUTE 
PRED_=EXP(a1*Age_Cracking*( 
100 + TMI)/100 ) - 1 . 
NLR Cracking_of_total_lane_area 
 
/OUTFILE='C:\Users\ranita.sen\A
ppData\Local\Temp\spss38652\S
PSSFNLR.TMP' 
 /PRED PRED_ 
 /CRITERIA SSCONVERGENCE 
1E-8 PCON 1E-8. 
Resources 
Processor time 
00:00:00.02 
Elapsed time 
00:00:00.02 
Files Saved 
Parameter estimates file 
C:\Users\ranita.sen\AppData\Loc
al\Temp\spss38652\SPSSFNLR.
TMP 


## Page 38
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 33 
 
Iteration Historyb 
Iteration numbera 
Residual sum of 
squares 
Parameter 
a1 
1.0 
19065.427 
.500 
1.1 
3299.734 
.375 
2.0 
3299.734 
.375 
2.1 
1486.207 
.273 
3.0 
1486.207 
.273 
3.1 
1392.655 
.228 
4.0 
1392.655 
.228 
4.1 
1392.638 
.227 
5.0 
1392.638 
.227 
5.1 
1392.638 
.227 
6.0 
1392.638 
.227 
6.1 
1392.638 
.227 
Derivatives are calculated numerically. 
a. Major iteration number is displayed to the left of the 
decimal, and minor iteration number is to the right of the 
decimal. 
b. Run stopped after 12 model evaluations and 6 derivative 
evaluations because the relative reduction between 
successive residual sums of squares is at most SSCON = 
1.00E-008. 
Parameter Estimates 
Parameter 
Estimate 
Std. error 
95% Confidence interval 
Lower bound 
Upper bound 
a1 
.227 
.014 
.201 
.254 
ANOVAa 
Source 
Sum of squares 
df 
Mean Squares 
Regression 
594.710 
1 
594.710 
Residual 
1392.638 
255 
5.461 
Uncorrected Total 
1987.347 
256 
Corrected Total 
1317.200 
255 
Dependent variable: Cracking(%_of_total_lane_area)a 
a. R squared = 1 - (Residual sum of squares) / (Corrected sum of squares) = .. 
 


## Page 39
Austroads Road Deterioration Model Update: Cracking 
 
 
 
 
Austroads 2023 | page 34 
 
 
