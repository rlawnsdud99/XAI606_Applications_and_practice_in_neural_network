XAI606 Project I
----------------------------------------------
### I. Project title
Detecting Confused Students using electroencephalogram (EEG) data from 10 college students while they watched MOOC video clips. 

      
### II. Project introduction
**Objective**   
The main goal of this project is to optimize the classification accuracy of predicting confused students using EEG signals.   
You can also conduct the further studies from a different perspective, e.g., analyzing the data with the frequencies or predicting the Mediation/Attention of the person.
   
**Motivation**    
Online education has emerged as an important educational medium during the COVID-19 pandemic. Despite the advantages of online education, it lacks face-to-face settings, which makes it very difficult to analyze the students’ level of interaction, understanding, and confusion. This project makes use of EEG data for student confusion detection for the online course platform.   

### III. Dataset description   
- Prepared videos are such as videos of the introduction of basic algebra or geometry, or the videos of Quantum Mechanics, and Stem Cell Research. Each video was about 2 minutes long and chopped the two-minute clip in the middle of a topic to make the videos more confusing.   
- The students wore a single-channel wireless MindSet that measured activity over the frontal lobe. The MindSet measures the voltage between an electrode resting on the forehead and two electrodes (one ground and one reference) each in contact with an ear.   
- After each session, the student rated his/her confusion level on a scale of 1-7, where one corresponded to the least confusing and seven corresponded to the most confusing. These labels if further normalized into labels of whether the students are confused or not. **This label is offered as 'Label' column.** (label 0: not confused, label 1: confused)
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
