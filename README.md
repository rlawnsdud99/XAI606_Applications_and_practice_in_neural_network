XAI606 Project I
----------------------------------------------
### I. Project title
Detecting Confused Students using EEG signal data from 10 college students while they watched MOOC video clips. 

      
### II. Project introduction
**Objective**   
The main goal of this project is to optimize the classification accuracy of EEG signals 
enhance the classification accuracy of EMG signals during static hand gestures. We'll employ advanced neural network architectures and machine learning techniques for this.       
   
**Motivation**    
Improving the accuracy of EMG-based gesture recognition has significant implications for BCI applications, including assistive technologies and human-computer interaction.

### III. Dataset description   
- Prepared videos are such as videos of the introduction of basic algebra or geometry, or the videos of Quantum Mechanics, and Stem Cell Research. Each video was about 2 minutes long and chopped the two-minute clip in the middle of a topic to make the videos more confusing.   
- The students wore a single-channel wireless MindSet that measured activity over the frontal lobe. The MindSet measures the voltage between an electrode resting on the forehead and two electrodes (one ground and one reference) each in contact with an ear.   
- After each session, the student rated his/her confusion level on a scale of 1-7, where one corresponded to the least confusing and seven corresponded to the most confusing. These labels if further normalized into labels of whether the students are confused or not. **This label is offered as 'Label' column of the .csv files.**
  - #### Columns:   
    -&nbsp;SubjectID   
    -&nbsp;VideoID   
    -&nbsp;Attention   
    -&nbsp;Mediation   
    -&nbsp;Raw   
    -&nbsp;Delta   
    -&nbsp;Theta   
    -&nbsp;Alpha1   
    -&nbsp;Alpha2   
    -&nbsp;Beta1   
    -&nbsp;Beta2   
    -&nbsp;Gamma1   
    -&nbsp;Gamma2   
    -&nbsp;Label
