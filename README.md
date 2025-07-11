# JFlow  

**JFlow** is a memory-efficient deep learning library written in Java, designed for both educational use and real-world machine learning applications. It provides low-level control over model training and supports limited hardware.  

![Training GIF](readme-images/Training-Example.gif)  

### Strengths  

#### 🧠 Memory Optimization  
- **Memory Options**: Train large models on large datasets with limited resources.  
    - Example: **CIFAR-10 CNN**, 1M parameters, < 2GB RAM (with image caching disabled)  
    - Example: **GPT-2 Small**, 124M parameters, < 9GB RAM for a sequence length of 512 (with gradient storage disabled)  
#### 🔍 Low-level Control & Debugging  
- Clean, Keras-similar UI for model training.  
- Implement custom training steps easily.  
- Debug mode for inspecting gradients.  
  
Simple Neural Network Debug    
![Gradient Debug Example](readme-images/Debug-Example.png)  

### Key Features  

#### Dataloader  
- Load images from CSV or directory.  
- Train-test-val-split and data batching.  

#### Transform  
- Normalize and augment images with built-in functions.  

#### Sequential Model  
![Model Summary](readme-images/Summary-Example.png)  
- Build models with a simple UI.  
    - High-level functions: train, predict.  
    - Low-level functions: forward(data), backward(data).  
- Save and load model weights.  

#### Supported Layers  
- **Dense**  
- **Conv2D**  (Under development)  
- **MaxPool2D**  
- **GlobalAveragePooling2D**  
- **Upsampling2D**  
- **BatchNorm**  
- **LayerNorm**  
- **Flatten**  
- **Embedding**  
- **Dropout**  

#### Supported Activation Functions  
- **ReLU**, **LeakyReLU**, **Sigmoid**, **Tanh**, **Softmax**, **Swish**, **Mish**, **GELU**.  
- **Custom Activation**: Easy to implement.  

#### Supported Optimizers  
- **SGD**, **AdaGrad**, **RMSprop**, **Adam**.  

#### Utilities  
- Plot images and confusion matrices.  
- JMatrix data type for low-level matrix operations.  

