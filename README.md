### Project Description

#### **Code Functionality**
- **(tea)train pseudo label generation model**  
  This script is used to train the teacher model for pseudo-label generation. The teacher model generates high-quality pseudo-labels to guide the training of the student model.

- **(stu)train final model**  
  This script is used to train the final student model. The student model leverages the pseudo-labels generated by the teacher model to achieve better performance.

- **PCS**  
  PCS (Pseudo Label Generation Code) is used to generate pseudo-labels. It produces pseudo-labels for training the student model based on the predictions from the teacher model.
