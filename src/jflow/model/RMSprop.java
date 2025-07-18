package jflow.model;

import java.util.HashMap;
import java.util.Map;
import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class RMSprop extends Optimizer<RMSprop> {
    private double learningRate;
    private double decay = 0.9;  // Decay rate for accumulated squared gradients
    private double epsilon = 1e-8;  // Small constant for numerical stability
    private double momentum = 0.0;  // Optional momentum parameter

    public RMSprop(double learningRate) {
        super("rmsprop");
        this.learningRate = learningRate;
    }

    public RMSprop(double learningRate, double decay, double epsilon) {
        super("rmsprop");
        this.learningRate = learningRate;
        this.decay = decay;
        this.epsilon = epsilon;
    }

    public RMSprop(double learningRate, double decay, double epsilon, double momentum) {
        super("rmsprop");
        this.learningRate = learningRate;
        this.decay = decay;
        this.epsilon = epsilon;
        this.momentum = momentum;
    }

    @Override
    public void applyUpdates(HashMap<String, JMatrix[]> layerGradients) {
        for (Map.Entry<String, JMatrix[]> entry : layerGradients.entrySet()) {
            TrainableLayer layer = getLayerID().get(entry.getKey());
            JMatrix[] gradients = entry.getValue();
            JMatrix[] moments = getMoments().get(layer);
            JMatrix[] updates = new JMatrix[gradients.length];

            for (int i = 0; i < gradients.length; i++) {
                JMatrix weightGradients = gradients[i];
                
                // Get accumulated squared gradients
                JMatrix accumSqGrad = moments[i * (momentum > 0 ? 2 : 1)];
                
                // Update accumulated squared gradients
                accumSqGrad.multiplyInPlace(decay)
                           .addInPlace(weightGradients.multiply(weightGradients).multiply(1 - decay));
                
                if (momentum > 0) {
                    // Get velocity matrix for momentum
                    JMatrix velocity = moments[i * 2 + 1];
                    
                    // Calculate parameter update with momentum
                    JMatrix paramUpdate = weightGradients.divide(accumSqGrad.sqrt().addInPlace(epsilon))
                                                         .multiplyInPlace(learningRate);
                    
                    // Update velocity
                    velocity.multiplyInPlace(momentum).addInPlace(paramUpdate);
                    updates[i] = velocity.copy();
                } else {
                    // Standard RMSprop update without momentum
                    updates[i] = weightGradients.divide(accumSqGrad.sqrt().addInPlace(epsilon))
                                               .multiplyInPlace(learningRate);
                }
            }
            
            // Apply updates
            layer.updateParameters(updates);
            // Reset gradients
            for (JMatrix m : layer.getParameterGradients()) {
                m.fill(0);
            }
        }
    }

    @Override
    public void initializeLayer(TrainableLayer layer) {
        JMatrix[] parameters = layer.getParameterGradients();
        
        // Without momentum: one matrix per parameter
        // With momentum: two matrices per parameter
        JMatrix[] moments = new JMatrix[parameters.length * (momentum > 0 ? 2 : 1)];
        
        for (int i = 0; i < parameters.length; i++) {
            // Initialize accumulated squared gradients matrix with zeros
            moments[i * (momentum > 0 ? 2 : 1)] = parameters[i].zerosLike();
            
            if (momentum > 0) {
                // Initialize velocity matrix with zeros if using momentum
                moments[i * 2 + 1] = parameters[i].zerosLike();
            }
        }
        
        getMoments().put(layer, moments);
        getLayerID().put(layer.getName(), layer);
    }
}