package jflow.layers.templates;

import jflow.data.JMatrix;

public abstract class NormalizationLayer<T extends NormalizationLayer<T>> extends TrainableLayer {
    private JMatrix gamma;
    private JMatrix beta;
    private JMatrix dGamma;
    private JMatrix dBeta;

    private double betaValue = 0;
    private double gammaValue = 1.0;

    private double epsilon = 1e-5;
    
    public NormalizationLayer(String type) {
        super(type);
    }

    /**
     * Set the initialization value of the gamma parameters. <p>
     * Default value: 1.0
     * @param value the initialization value -> {@code gamma.fill(value)}
     */
    @SuppressWarnings("unchecked")
    public T gammaValue(double value) {
        gammaValue = value;
        return (T) this;
    }

    /**
     * Set the initialization value of the beta parameters. <p>
     * Default value: 0.0
     * @param value the initialization value -> {@code beta.fill(value)}
     */
    @SuppressWarnings("unchecked")
    public T betaValue(double value) {
        betaValue = value;
        return (T) this;
    }

    /**
     * Declare epsilon for this layer. <p>
     * Default value: 1e-5.
     * @param epsilon the epsilon value to use.
     */
    @SuppressWarnings("unchecked")
    public T withEpsilon(double epsilon) {
        this.epsilon = epsilon;
        return (T) this;
    }
    /**
     * Access gamma for use in the child class.
     */
    protected JMatrix getGamma() {
        return gamma;
    }

    /**
     * Access beta for use in the child class.
     */
    protected JMatrix getBeta() {
        return beta;
    }

    /**
     * Access dGamma for use in the child class.
     */
    protected JMatrix getDGamma() {
        return dGamma;
    }

    /**
     * Access dBeta for use in the child class.
     */
    protected JMatrix getDBeta() {
        return dBeta;
    }

    /**
     * Get epsilon for use in the child class.
     */
    protected double getEpsilon() {
        return epsilon;
    }

    /**
     * Return this normalization layer's running stats 
     * to enable serialization. If none, return null.
     */
    protected abstract JMatrix[] getRunningStats();

    /** 
     * Declare the shape of gamma and beta.
     */
    protected abstract int[] parameterShape(int[] inputShape);

    @Override
    public void build(int[] inputShape) {
        int[] initShape = parameterShape(inputShape);

        // Initialize parameters
        gamma = JMatrix
            .zeros(initShape)
            .fill(gammaValue)
            .label("gamma");
        beta = JMatrix
            .zeros(initShape)
            .fill(betaValue)
            .label("beta");

        dGamma = JMatrix
            .zeros(initShape)
            .label("dGamma");
        dBeta = JMatrix
            .zeros(initShape)
            .label("dBeta");

        // Add gamma and beta to the param count
        setNumTrainableParameters(gamma.size() + beta.size());
    }


    @Override
    public JMatrix[] getParameters() {
        JMatrix[] runningStats = getRunningStats();
        if (runningStats == null) {
            return new JMatrix[]{gamma, beta};
        }
        int numSerializable = runningStats.length + 2;
        JMatrix[] serializable = new JMatrix[numSerializable];
        serializable[0] = gamma;
        serializable[1] = beta;
        for (int i = 2; i < numSerializable; i++) {
            serializable[i] = runningStats[i - 2];
        }        

        return serializable;
    }

    @Override
    public JMatrix[] getParameterGradients() {
        return new JMatrix[] {dGamma, dBeta};
    }

    @Override
    public void updateParameters(JMatrix[] parameterUpdates) {
        gamma.subtractInPlace(parameterUpdates[0]);
        beta.subtractInPlace(parameterUpdates[1]);
    }
}