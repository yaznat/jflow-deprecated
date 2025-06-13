package jflow.model;

import java.util.HashMap;
import java.util.Map;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class SGD extends Optimizer {
    private double learningRate;
    private double momentum = 0.0; // Default to vanilla SGD (no momentum)
    private boolean useNesterov = false;

    protected SGD(double learningRate) {
        super("sgd");
        this.learningRate = learningRate;
    }

    protected SGD(double learningRate, double momentum, boolean useNesterov) {
        super("sgd");
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.useNesterov = useNesterov;
    }

    @Override
    public void apply(HashMap<String, JMatrix[]> layerGradients) {
        boolean needsClipping = false;
        double clipScale = 1.0;
        if (useClipping()) {
            double globalGradNormSquared = 0.0;
            for (JMatrix[] grads : layerGradients.values()) {
                for (JMatrix grad : grads) {
                    double frobeniusNorm = grad.l2Norm();
                    globalGradNormSquared += frobeniusNorm * frobeniusNorm;
                }
            }

            double globalGradNorm = Math.sqrt(globalGradNormSquared);
            
            if (useClipping() && globalGradNorm > getClipNorm()) {
                clipScale = getClipNorm() / (globalGradNorm + 1e-6);  // epsilon for numerical stability
                needsClipping = true;
            }
        }
        for (Map.Entry<String, JMatrix[]> entry : layerGradients.entrySet()) {
            TrainableLayer layer = getLayerID().get(entry.getKey());
            JMatrix[] gradients = entry.getValue();
            JMatrix[] updates = new JMatrix[gradients.length];

            
            if (momentum > 0) {
                // SGD with momentum
                JMatrix[] velocities = getMoments().get(layer);
                
                for (int i = 0; i < gradients.length; i++) {
                    JMatrix weightGradients = gradients[i];
                    JMatrix velocity = velocities[i];

                    // Clip if needed
                    if (needsClipping) {
                        weightGradients.multiplyInPlace(clipScale);
                    }
                    
                    // Update velocity (momentum term)
                    velocity.multiplyInPlace(momentum).addInPlace(weightGradients.multiply(learningRate));
                    
                    if (useNesterov) {
                        // Nesterov accelerated gradient
                        updates[i] = weightGradients.multiply(learningRate).addInPlace(velocity.multiply(momentum));
                    } else {
                        // Standard momentum
                        updates[i] = velocity;
                    }
                }
            } else {
                // Vanilla SGD (no momentum)
                for (int i = 0; i < gradients.length; i++) {
                    // Clip if needed
                    if (needsClipping) {
                        gradients[i].multiplyInPlace(clipScale);
                    }
                    updates[i] = gradients[i].multiply(learningRate);
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
    protected void initializeLayer(TrainableLayer layer) {
        JMatrix[] gradients = layer.getParameterGradients();
        int numWeights = gradients.length;

        JMatrix[] moments = new JMatrix[numWeights];

        for (int i = 0; i < numWeights; i++) {
            // Initialized to zero
            JMatrix vWeights = gradients[i].zerosLike();
            moments[i] = vWeights;
        }

        getMoments().put(layer, moments);
        getLayerID().put(layer.getName(), layer);
    }
}