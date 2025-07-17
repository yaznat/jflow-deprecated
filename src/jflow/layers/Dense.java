package jflow.layers;

import jflow.data.JMatrix;
import jflow.layers.templates.ParametricLayer;

public class Dense extends ParametricLayer<Dense> {
    private JMatrix weights;
    private JMatrix dWeights;
    private JMatrix biases;
    private JMatrix dBiases;

    private final int outputSize;
    private final boolean useBias;

    private boolean scaleDuringMatmul = true;
    private float[] tiedWeights = null;
  
    /**
     * Represents a fully connected layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Dense(...)} instead of {@code new Dense(...)}.
     */
    public Dense(int size, int[] inputShape, boolean useBias) {
        super("dense");
        this.outputSize = size;
        this.useBias = useBias;
        setInputShape(inputShape);
    }

    /**
     * Indicate if scaling by 1/k should be applied during matrix multiplication.
     * Enabled by default. <p>
     * Recommended for image data.
     * Not recommended for transformers.
     * @param enabled Enable or disable scaling during matrix multiplication.
     */
    public Dense scaleDuringMatmul(boolean enabled) {
        scaleDuringMatmul = enabled;
        return this;
    }

    /**
     * Set the weight matrix for weight tying.
     * @param weights The weight matrix to use for this Dense layer.
     * @param removeFromParamCount Indicate if the weight matrix of this Dense 
     * layer should be removed from the parameter count.
     */
    public Dense weightTie(float[] weights, boolean removeFromParamCount) {
        this.tiedWeights = weights;
        if (removeFromParamCount) {
            // Remove the matrix from numTrainableParameters
            setNumTrainableParameters((useBias) ? outputSize : 0);
        } else {
            setNumTrainableParameters((useBias) ? outputSize + weights.length : weights.length);
        }
        return this;
    }  

    @Override
    protected void build(int IDnum) {
        super.build(IDnum);
        int inputSize;
        int[] inputShape = getInputShape();

        if (inputShape == null) {
            throw new IllegalStateException(
                    "In " + this.getClass().getSimpleName() + 
                    ": Cannot build the first layer without an input shape."
                );
        } else {
            inputSize = inputShape[1]; // Feature dimension
        }

        initParams(inputSize);

        if (tiedWeights == null) {
            int numTrainableParameters = 0;
            numTrainableParameters += inputSize * outputSize;
            if (useBias) {
                numTrainableParameters += outputSize;
            }
            setNumTrainableParameters(numTrainableParameters);
        } else {
            try {
                weights.setMatrix(tiedWeights);
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                    "Invalid matrix size for weight tying. Expected: "
                    + weights.size() + " Got: " + tiedWeights.length
                );
            }
            tiedWeights = null;
        }
    }

    private void initParams(int inputSize) {
        // Check for user-specified initialization
        if (useCustomInit()) {
            weights = initCustomWeight(inputSize, outputSize, 1, 1);
            if (useBias) {
                biases = initCustomWeight(1, outputSize, 1, 1);
            }
        } else {
            // Use standard he initialization
            double stddev = Math.sqrt(2.0 / inputSize);
            weights = JMatrix.normal(
                inputSize, outputSize, 1, 1,
                0.0, stddev
            );
            if (useBias) {
                biases = JMatrix.normal(
                    1, outputSize, 1, 1,
                    0.0, stddev
                );
            }
        }
        weights.label("weights");

        dWeights = JMatrix.zeros(
            inputSize, outputSize, 1, 1
        )
        .label("dWeights");

        if (useBias) {
            biases.label("biases");
            dBiases = JMatrix.zeros(
                outputSize, 1, 1, 1
            )
            .label("dBiases");
        }   
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        // Cache the input for backpropagation
        cacheInput(input, training);

        // Calculate forward output
        JMatrix output = input.matmul(weights, scaleDuringMatmul); 

        if (useBias) {
            // Broadcast add: (1,F,1,1) over (N,F,1,1)
            output.addInPlace(biases);
        }
        
        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        // Calculate dWeights and dBiases
        JMatrix weightGrad = getLastInput().T().matmul(gradient, scaleDuringMatmul);
        dWeights.addInPlace(weightGrad); // Accumulate updates

        if (useBias) {
            JMatrix biasGrad = gradient.T().sum(0).multiplyInPlace(1.0 / gradient.shape(1)); // Scaled sum
            dBiases.addInPlace(biasGrad); // Accumulate updates
        }

        // Normalize dWeights and dBiases
        adaptiveGradientClip(weights, biases, dWeights, dBiases, 1e-2);

        // Calculate loss w.r.t previous layer
        JMatrix dX = gradient.matmul(weights.T(), scaleDuringMatmul);
       
        return trackGradient(dX);
    }

    // Adaptively clip with L2 norm
    private void adaptiveGradientClip(
        JMatrix weights, 
        JMatrix biases, 
        JMatrix dWeights, 
        JMatrix dBiases, 
        double epsilon
    ) {
        // Clip weights
        double weightNorm = weights.l2Norm();
        double gradWeightNorm = dWeights.l2Norm();

        double maxWeightNorm = Math.max(gradWeightNorm, epsilon * weightNorm);
        
        if (gradWeightNorm > maxWeightNorm) {
            double weightScale = maxWeightNorm / gradWeightNorm;
            dWeights.multiplyInPlace(weightScale);
        }
        
        // Clip biases
        if (useBias) {
            double biasNorm = biases.l2Norm();
            double gradBiasNorm = dBiases.l2Norm();

            double maxBiasNorm = Math.max(gradBiasNorm, epsilon * biasNorm);
            
            if (gradBiasNorm > maxBiasNorm) {
                double biasScale = maxBiasNorm / gradBiasNorm;
                dBiases.multiplyInPlace(biasScale);
            }
        }
    }

    @Override
    public JMatrix[] getParameters() {
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
    public void updateParameters(JMatrix[] parameterUpdates) {
        weights.subtractInPlace(parameterUpdates[0]);
        if (useBias) {
            biases.subtractInPlace(parameterUpdates[1]);
        }
    }

    @Override
    public int[] outputShape() {
        int[] outputShape = new int[] {1, outputSize};
        return outputShape;
    }
}