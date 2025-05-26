# JFlow  

**JFlow** is a memory-efficient deep learning library designed in Java for both educational purposes and real-world machine learning applications. It provides low-level control over model training and supports limited hardware.  

![Alt text](readme-images/Callback-Example.png)  

### Strengths  

#### Memory Optimization  
- **Low Memory Mode**: Train large models with limited resources (ideal for datasets larger than system memory).  
    - Example: **CIFAR-10 CNN**, 1M parameters, < 2GB RAM (with low memory mode enabled)  
    - Example: **GPT-2 Small**, 124M parameters, < 9GB RAM for a sequence length of 512 (with gradient storage disabled)  
#### Performance Benchmarks

##### Training GPT-2 Small (124M parameters) - Apple M4 Max  
- **Allocated 9GB RAM:** ~480,000 tokens/hour  
- **Allocated 16GB RAM:** ~550,000 tokens/hour  
**Note:** Performance differences are due to Java memory allocation overhead, not low memory mode. Disabling gradient storage reduces memory usage but does not impact speed.  
#### Low-level Control & Debugging  
- Clean, Keras-similar UI for model training.  
- Implement custom training steps easily.  
- Debug mode for inspecting gradients.  

### Key Features  

#### Dataloader  
- Load images from CSV or directory.  
- Train-test-split and data batching.  

#### Transform  
- Normalize and augment images with built-in functions.  

#### Sequential Model  
- Build models with a simple UI.  
    - High-level functions: train, predict.  
    - Low-level functions: forward(data), backward(data).  
- Save and load model weights.  

#### Supported Layers  
- **Dense**  
- **Conv2D**  
- **MaxPool2D**  
- **Upsampling2D**  
- **BatchNorm**  
- **LayerNorm**  
- **Flatten**  
- **GlobalAveragePooling2D**  
- **Embedding**  

#### Supported Activation Functions  
- **ReLU**, **LeakyReLU**, **Sigmoid**, **Tanh**, **Softmax**, **Swish**, **Mish**, **GELU**.  
- **Custom Activation**: Easy to implement.  

#### Supported Optimizers  
- **SGD**, **AdaGrad**, **RMSprop**, **Adam**.  

#### Utilities  
- Plot images and confusion matrices.  
- JMatrix data type for low-level matrix operations.  

