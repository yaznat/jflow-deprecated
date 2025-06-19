package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class Dense extends TrainableLayer {
    private JMatrix weights;
    private JMatrix dWeights;
    private JMatrix lastInput;
    private JMatrix biases;
    private JMatrix dBiases;

    private float[] tiedWeights = null;

    private int outputSize;
    private double customInitScale;

    private boolean useBias = true;
    private boolean useCustomScale = false;


    /**
     * Represents a fully connected layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Dense(...)} instead of {@code new Dense(...)}.
     */
    public Dense(int size) {
        super("dense");
        this.outputSize = size;
    }

    /**
     * Represents a fully connected layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Dense(...)} instead of {@code new Dense(...)}.
     */
    public Dense(int size, boolean useBias) {
        this(size);
        this.useBias = useBias;
    }

    /**
     * Represents a fully connected layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Dense(...)} instead of {@code new Dense(...)}.
     */
    public Dense(int size, int[] inputShape) {
        this(size);

        if (inputShape.length != 2) { // + 1 for internal batch dimension
            throw new IllegalArgumentException(
                "Dense input shape should have 1 dimension. Got: "
                + (inputShape.length - 1) + "."
            );
        }
        setInputShape(inputShape);
    }

    /**
     * Represents a fully connected layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Dense(...)} instead of {@code new Dense(...)}.
     */
    public Dense(int size, int[] inputShape, boolean useBias) {
        this(size, inputShape);
        this.useBias = useBias;
    }

    /**
     * Initialize the weights of this Dense layer with a custom scale.
     * @param scale The magnitude of initialization. 
     * Parameters will be in the range [-scale, scale].
     */
    public Dense customInitScale(double scale) {
        useCustomScale = true;
        customInitScale = scale;
        return this;
    }

    /**
     * Set the weight matrix for weight tying.
     * @param weights The weight matrix to use for this Dense layer.
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
            inputSize = inputShape[1]; // Channel dimension
        }

        // Xavier initialization or custom
        double scale = (useCustomScale) ? customInitScale : Math.sqrt(2.0 / (inputSize + outputSize));

        double min = -1.0 * scale;
        double max = scale;

        weights = JMatrix.uniform(outputSize, inputSize, 1, 1, min, max).setName("weights");
        dWeights = JMatrix.zeros(outputSize, inputSize, 1, 1).setName("dWeights");

        if (useBias) {
            biases = JMatrix.uniform(outputSize, 1, 1, 1, min, max).setName("biases");
            dBiases = JMatrix.zeros(outputSize, 1, 1, 1).setName("dBiases");
        }   

        if (tiedWeights == null) {
            int numTrainableParameters = 0;
            numTrainableParameters += inputSize * outputSize;
            if (useBias) {
                numTrainableParameters += outputSize;
            }
            setNumTrainableParameters(numTrainableParameters);
        } else {
            weights.setMatrix(tiedWeights);
            tiedWeights = null;
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
        JMatrix output = weights.matmul(input, true); // Scaled matmul product

        if (useBias) {
            applyBiasByRow(output, biases); 
        }
    
        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        // Transpose if necessary
        if (gradient.channels() * gradient.height() * gradient.width() != 
                lastInput.channels() * lastInput.height() * lastInput.width()) {
            gradient = gradient.transpose2D();
        }
        // Calculate dWeights and dBiases
        JMatrix weightGrad = gradient.matmul(lastInput.transpose2D(), true); // Scaled matmul product
        dWeights.setMatrix(weightGrad.unwrap()); // Avoid reassigning reference

        if (useBias) {
            JMatrix biasGrad = gradient.sum(0).multiplyInPlace(1.0 / gradient.length()); // Scaled sum
            dBiases.setMatrix(biasGrad.unwrap()); // Avoid reassigning reference
        }

        // Free memory
        lastInput = null;
       
        // Normalize dWeights and dBiases
        adaptiveGradientClip(weights, biases, dWeights, dBiases, 1e-2);

        // Calculate loss w.r.t previous layer
        JMatrix dX = weights.transpose2D().matmul(gradient, true); // Scaled matmul product

        return trackGradient(dX);
    }

    @Override
    public void updateParameters(JMatrix[] parameterUpdates) {
        weights.subtractInPlace(parameterUpdates[0]);
        if (useBias) {
            biases.subtractInPlace(parameterUpdates[1]);
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
    public int[] outputShape() {
        int[] outputShape = new int[] {1, outputSize};
        return outputShape;
    }

    private void applyBiasByRow(JMatrix output, JMatrix bias) {
        int rows = output.length();
        int cols = output.channels() * output.height() * output.width();
        
        IntStream.range(0, rows).parallel().forEach(i -> {
            for (int j = 0; j < cols; j++) { 
                output.set(i * cols + j, output.get(i * cols + j) + bias.get(i));
            }
        });
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
}



