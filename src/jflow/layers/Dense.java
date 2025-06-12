package jflow.layers;

import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class Dense extends TrainableLayer {
    private JMatrix weights;
    private JMatrix dWeights;
    private JMatrix lastInput;
    private JMatrix biases;
    private JMatrix dBiases;

    private int outputSize;

    private boolean useBias = true;

    public Dense(int size, int[] inputShape, boolean useBias) {
        this(size, inputShape);
        this.useBias = useBias;
    }

    public Dense(int size, int[] inputShape) {
        this(size);

        if (inputShape.length != 1) {
            throw new IllegalArgumentException(
                "Dense input shape should have 1 dimension. Got: "
                + inputShape.length + "."
            );
        }
        setInputShape(inputShape);

    }

    public Dense(int size) {
        super("dense");
        this.outputSize = size;
    }

    public Dense(int size, boolean useBias) {
        this(size);
        this.useBias = useBias;
    }

    @Override
    public void build(int IDnum) {
        super.build(IDnum);
        int inputSize;
        if (internalGetInputShape() != null) {
            inputSize = internalGetInputShape()[0];
        } else {
            if (getPreviousLayer() == null) {
                throw new IllegalStateException(
                    "In " + this.getClass().getSimpleName() + 
                    ": Cannot build the first layer without an input shape."
                );
            }
            // Channel dimension
            inputSize = getPreviousLayer().outputShape()[1];
        }
        int numTrainableParameters = inputSize * outputSize;
        if (useBias) {
            numTrainableParameters += outputSize;
        }
        setNumTrainableParameters(numTrainableParameters);


        // Initialize weights and biases
        float[] weights = new float[outputSize * inputSize];
        float[] biases = new float[outputSize];

        // He Initialization
        double scale = Math.sqrt(2.0 / inputSize);

        IntStream.range(0, outputSize).parallel().forEach(i -> {
            for (int j = 0; j < inputSize; j++) {
                weights[i * inputSize + j] = (float)(
                    (ThreadLocalRandom.current().nextDouble() - 0.5) * scale);  
            }
            if (useBias) {
                biases[i] = (float)((Math.random() - 0.5) * 0.5);
            }
            
        });

        this.weights = JMatrix.wrap(weights, outputSize, inputSize, 1, 1).setName("weights");
        this.dWeights = JMatrix.zeros(outputSize, inputSize, 1, 1).setName("dWeights");

        if (useBias) {
            this.biases = JMatrix.wrap(biases, outputSize, 1, 1, 1).setName("biases");
            this.dBiases = JMatrix.zeros(outputSize, 1, 1, 1).setName("dBiases");
        }   
        
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        if (input.length() != 
                weights.channels() * weights.height() * weights.width()) {
            input = input.transpose2D();
        }
        // Store lastInput for backpropagation
        if (training) {
            lastInput = input;
        }

        // Calculate forward output
        JMatrix A = weights.matmul(input, true); // scaled

        if (useBias) {
            applyBiasByRow(A, biases); 
        }
       

        return trackOutput(A, training);
    }
    @Override
    public JMatrix backward(JMatrix gradient) {

        // Calculate dWeights and dBiases
        if (gradient.channels() * gradient.height() * gradient.width() != 
                lastInput.channels() * lastInput.height() * lastInput.width()) {
            gradient = gradient.transpose2D();
        }
        dWeights.setMatrix(gradient.matmul(lastInput.transpose2D(), true).getMatrix()); // avoid reassigning reference
        if (useBias) {
            dBiases.setMatrix(gradient.sum0(false));
        }

        // Save memory
        lastInput = null;
       

        // Normalize dWeights and dBiases
        adaptiveGradientClip(weights, biases, dWeights, dBiases, 1e-2);

        // Calculate loss w.r.t previous layer
        JMatrix dX = weights.transpose2D().matmul(gradient, true); // Scaled matmul product

        // float normValue = dX.frobeniusNorm();

        // // Apply gradient scaling if the norm exceeds the threshold
        // float threshold = 2.0f;
        // if (normValue > threshold) {
        //     // Scale the gradients
        //     dX.multiplyInPlace(threshold / normValue);
        // }

        return trackGradient(dX);
    }

    @Override
    public void updateParameters(JMatrix[] parameterUpdates) {
        weights.subtractInPlace(parameterUpdates[0]);
        if (useBias) {
            biases.subtractInPlace(parameterUpdates[1]);
        }
    }

    private void applyBiasByRow(JMatrix A, JMatrix bias) {
        int rows = A.length();
        int cols = A.channels() * A.height() * A.width();
        
        IntStream.range(0, rows).parallel().forEach(i -> {
            for (int j = 0; j < cols; j++) { 
                A.set(i * cols + j, A.get(i * cols + j) + bias.get(i));
            }
        });
    }
    
    // Adaptively clip with frobenius norm
    private void adaptiveGradientClip(JMatrix weights, JMatrix biases, JMatrix dWeights, JMatrix dBiases, double epsilon) {
        // Clip weights
        double weightNorm = weights.frobeniusNorm();
        double gradWeightNorm = dWeights.frobeniusNorm();
        double maxWeightNorm = Math.max(gradWeightNorm, epsilon * weightNorm);
        
        if (gradWeightNorm > maxWeightNorm) {
            double scaleWeight = maxWeightNorm / gradWeightNorm;
            dWeights.multiplyInPlace(scaleWeight);
        }
        
        // Clip biases
        if (useBias) {
            double biasNorm = biases.frobeniusNorm();
            double gradBiasNorm = dBiases.frobeniusNorm();
            double maxBiasNorm = Math.max(gradBiasNorm, epsilon * biasNorm);
            
            if (gradBiasNorm > maxBiasNorm) {
                double scaleBias = maxBiasNorm / gradBiasNorm;
                dBiases.multiplyInPlace(scaleBias);
            }
        }
    }
    /**
     * Set the weight matrix for weight tying.
     * @param weights a reference to the weight matrix to use for this Dense layer.
     */
    public void weightTie(float[] weights) {
        this.weights.setMatrix(weights);
        // Remove the matrix from numTrainableParameters
        setNumTrainableParameters((useBias) ? outputSize : 0);
    }

    @Override
    public JMatrix[] getWeights() {
        if (useBias) {
            return new JMatrix[]{weights, biases};
        } else {
            return new JMatrix[]{weights};
        }
        
    }

    @Override
    public JMatrix[] getParameterGradients() {
        if (useBias) {
            return new JMatrix[]{dWeights, dBiases};
        } else {
            return new JMatrix[]{dWeights};
        }
    }

    @Override
    public int[] outputShape() {
        int[] outputShape = new int[] {-1, outputSize};
        return outputShape;
    }

  
}



