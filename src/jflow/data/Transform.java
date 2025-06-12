package jflow.data;
import java.util.ArrayList;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

public class Transform {
    // Store transforms
    private ArrayList<Function<JMatrix, JMatrix>> transforms;
    // Keep track of image bounds for clipping
    private int[] normBounds = new int[2];
    /**
     * Initializes an empty Transform.
     */
    public Transform() {
        transforms = new ArrayList<>();
    }

    protected ArrayList<Function<JMatrix, JMatrix>> getTransforms() {
        return transforms;
    }

    // Provide information about 
    protected int[] normBounds() {
        return normBounds;
    }

    /**
     * Normalize image data to [0,1].
     */
    public Transform normalizeSigmoid() {
        normBounds[0] = 0;
        normBounds[1] = 1;
        transforms.add(
            image -> {
                return image.multiply(1.0 / 255);

            }
        );
        return this;
    }

   /**
     * Normalize image data to [-1,1].
     */
    public Transform normalizeTanh() {
        normBounds[0] = -1;
        normBounds[1] = 1;
        transforms.add(
            image -> {
                return image.multiply(1 / 127.5).subtractInPlace(1);
            }
        );
        return this;
    }

    /**
     * Polarize grayscale pixel values to 0 (black) and 255 (white) only.
     */
    public Transform grayscaleFullContrast() {
        transforms.add(
            image -> {
                int channels = image.channels();
                int height = image.height();
                int width = image.width();
        
                JMatrix contrasted = image.zerosLike();

                for (int c = 0; c < channels; c++) {
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            contrasted.set(0, c, i, j, (image.get
                                (0, c, i, j) > 0) ? 255 : 0);
                        }
                    }
                }
        
                return contrasted;
            }
        );
        return this;
    }

    /**
     * Invert RGB values (0 -> 255...)
     */
    public Transform invert() {
        transforms.add(
            image -> {
                int channels = image.channels();
                int height = image.height();
                int width = image.width();
        
                JMatrix inverted = image.zerosLike();
        
                for (int c = 0; c < channels; c++) {
                    for (int i = 0; i < height; i++) {
                        for (int j = 0; j < width; j++) {
                            inverted.set(0, c, i, j, 255 - image.get(0, c, i, j));
                        }
                    }
                }
        
                return inverted;
            }
        );
        return this;
    }

     /**
     * Rotate images by either 90, 180, or 270 degrees.
     */
    public Transform randomRotation() {
        System.out.println(normBounds[0]);
        transforms.add(
            image -> {
                int numRotations = ThreadLocalRandom.current().nextInt(4);
                for (int i = 0; i < numRotations; i++) {
                    image = image.transpose4D();
                }
                return image;
            }
        );
        return this;
    }
    /**
     * 50% chance to flip the image horizontally.
     */
    public Transform randomFlip() {
        transforms.add(
            image -> {
                if (ThreadLocalRandom.current().nextDouble() > 0.5) {
                    image = image.transpose4D().transpose4D();
                }
                return image;
            }
        );
        return this;
    }
    /**
     * Adds a random brightness value to images.
     */
    public Transform randomBrightness() {
        transforms.add(
            image -> {
                // Random value from -0.2 to 0.2
                double brightness = Math.random() / 2.5 - 0.2;
                return image
                    .add(brightness)
                    .clip(normBounds[0], normBounds[1]);
            }
        );
        return this;
    }

    /**
     * Resize with nearest-neighbor interpolation.
     */
    public Transform resize(int height, int width) {
        transforms.add(image -> {
            int channels = image.channels();
            int oldHeight = image.height();
            int oldWidth = image.width();
            
            JMatrix resized = JMatrix.zeros(1, channels, height, width);
    
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < height; i++) {
                    for (int j = 0; j < width; j++) {
                        int srcY = (int)((i / (float)height) * oldHeight);
                        int srcX = (int)((j / (float)width) * oldWidth);
                        srcY = Math.min(srcY, oldHeight - 1);
                        srcX = Math.min(srcX, oldWidth - 1);
                        
                        resized.set(1, c, i, j, image.get(1, c, srcY, srcX));
                    }
                }
            }
            
            return resized;
        });
        return this;
    }
    
    /**
     * Define a custom Java Function to add it to this Transform.
     * @param func a custom Java Function that accepts and returns a JMatrix.
     * The input JMatrix represents an image in the shape (N,C,H,W). The batch dimension 
     * (N) will be equal to 1.
     */
    public Transform customTransform(Function<JMatrix, JMatrix> func) {
        transforms.add(func);
        return this;
    }
}
