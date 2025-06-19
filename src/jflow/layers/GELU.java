package jflow.layers;

import java.util.stream.IntStream;
import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

/**
 * GELU (Gaussian Error Linear Unit) activation function.
 */
public class GELU extends ShapePreservingLayer {
    private JMatrix lastInput;
    
    public GELU() {
        super("gelu");
    }
    
    
    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        lastInput = input;
        int size = input.size();
        JMatrix output = input.zerosLike();
        
        IntStream.range(0, size).parallel().forEach(i -> {
            double x = input.get(i);
            double geluValue;
            
            // Exact formula: x * 0.5 * (1 + erf(x/sqrt(2)))
            geluValue = x * 0.5 * (1 + erf(x / Math.sqrt(2)));

            
            output.set(i, geluValue);
        });
        
        return trackOutput(output, training);
    }
    
    @Override
    public JMatrix backward(JMatrix gradient) {
        int size = gradient.size();
        JMatrix dZ = lastInput.zerosLike();
        
        IntStream.range(0, size).parallel().forEach(i -> {
            double x = lastInput.get(i);
            double derivative;
            
            // Derivative of GELU exact formula
            // 0.5 * (1 + erf(x/sqrt(2))) + x * 0.5 * (2/sqrt(2*pi)) * exp(-(x/sqrt(2))^2)
            double xOverSqrt2 = x / Math.sqrt(2);
            double erfTerm = 0.5 * (1 + erf(xOverSqrt2));
            double gaussianTerm = 0.5 * Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
            derivative = erfTerm + x * gaussianTerm;
            
            dZ.set(i, gradient.get(i) * derivative);
        });
        
        return trackGradient(dZ);
    }
    

    // The error function, erf(x), with a polynomial approximation
    private double erf(double x) {
        // Constants for Abramowitz and Stegun approximation
        final double a1 = 0.254829592;
        final double a2 = -0.284496736;
        final double a3 = 1.421413741;
        final double a4 = -1.453152027;
        final double a5 = 1.061405429;
        final double p = 0.3275911;
        
        // Save the sign of x
        int sign = (x < 0) ? -1 : 1;
        x = Math.abs(x);
        
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        
        return sign * y;
    }
}