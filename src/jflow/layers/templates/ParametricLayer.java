package jflow.layers.templates;

import jflow.data.JMatrix;

public abstract class ParametricLayer<T extends ParametricLayer<T>> extends TrainableLayer {
    private boolean initUniform = false;
    private boolean initNormal = false;
    private double[] initValues;

    public ParametricLayer(String type) {
        super(type);
    }

    /**
     * Initialize the weights of this layer in a uniform range.
     * @param min the minimum value.
     * @param max the maximum value.
     */
    @SuppressWarnings("unchecked")
    public T initUniform(double min, double max) {
        initUniform = true;
        initValues = new double[]{min, max};
        return (T) this;
    }

    /**
     * Initialize the weights of this layer in a normal distribution.
     * @param mean the mean of the distribution.
     * @param stddev the standard deviation of the distribution.
     */
    @SuppressWarnings("unchecked")
    public T initNormal(double mean, double stddev) {
        initNormal = true;
        initValues = new double[]{mean, stddev};
        return (T) this;
    }

    protected boolean useCustomInit() {
        return initUniform || initNormal;
    }
    
    protected JMatrix initCustomWeight(int length, int channels, int height, int width) {
        if (initNormal) {
            return JMatrix.normal(length, channels, height, width, initValues[0], initValues[1]);
        }
        if (initUniform) {
            return JMatrix.uniform(length, channels, height, width, initValues[0], initValues[1]);
        }
        throw new IllegalCallerException(
            "For developer: initCustomWeight() should not be called unless useCustomInit() is true."
        );
    }
}
