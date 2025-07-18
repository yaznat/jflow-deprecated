package jflow.layers;

import java.util.stream.IntStream;
import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class GELU extends ShapePreservingLayer {
    private static final double SQRT_2_OVER_PI = Math.sqrt(2.0 / Math.PI);
    private static final double COEFF = 0.044715;

    /**
     * The GELU activation.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code GELU()} instead of {@code new GELU()}.
     */
    public GELU() {
        super("gelu");
    }
    
    /*
     * Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
     * From the original GELU paper: https://arxiv.org/abs/1606.08415
     */ 
    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        cacheInput(input, training);
        int size = input.size();
        JMatrix output = input.zerosLike();

        IntStream.range(0, size).parallel().forEach(i -> {
            double x = input.get(i);
            double xCube = x * x * x;
            double inner = SQRT_2_OVER_PI * (x + COEFF * xCube);
            double tanhInner = Math.tanh(inner);
            output.set(i, 0.5 * x * (1 + tanhInner));
        });

        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        JMatrix x = getLastInput();
        JMatrix dZ = x.zerosLike();
        int size = gradient.size();

        IntStream.range(0, size).parallel().forEach(i -> {
            double xi = x.get(i);
            double x3 = xi * xi * xi;
            double inner = SQRT_2_OVER_PI * (xi + COEFF * x3);
            double tanhInner = Math.tanh(inner);
            double sech2 = 1 - tanhInner * tanhInner;

            double term1 = 0.5 * (1 + tanhInner);
            double term2 = 0.5 * xi * sech2 * SQRT_2_OVER_PI * (1 + 3 * COEFF * xi * xi);

            double grad = gradient.get(i) * (term1 + term2);
            dZ.set(i, grad);
        });

        return trackGradient(dZ);
    }
}