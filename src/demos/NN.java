package demos;
import jflow.data.*;
import jflow.model.*;
import jflow.utils.JPlot;
import jflow.utils.Metrics;

// Static import for cleaner UI
import static jflow.model.Builder.Dense;
import static jflow.model.Builder.GELU;
import static jflow.model.Builder.Dropout;
import static jflow.model.Builder.InputShape;
import static jflow.model.Builder.Softmax;
import static jflow.model.Builder.Adam;
import static jflow.model.Builder.ModelCheckpoint;

/**
 * Demo to train a neural network on the MNIST dataset.
 * Reaches ~97% test accuracy after 10 epochs.
 * 
 * Note: All library methods are documented via Javadoc. 
 * Hover in an IDE for details.
 */
public class NN {
    public static void main(String[] args) {
        final boolean DEBUG_ENABLED = false; // Enable full forward/backward tracing during training

        // training constants
        final int BATCH_SIZE = 64;
        final int NUM_EPOCHS = 10;
        final double VAL_PERCENT = 0.05;
        final double TEST_PERCENT = 0.05;
        
        // MNIST constants
        final int NUM_CLASSES = 10;
        final int FLAT_IMAGE_SIZE = 784;

        // Initialize the dataloader
        Dataloader loader = new Dataloader();

        // Load data
        loader.loadFromCSV("datasets/MNIST.csv", true, 1.0);

        // Prepare the transform and apply normalization
        Transform transform = new Transform()
            .normalizeSigmoid();

        loader.applyTransform(transform);

        // Prepare data for training
        loader.setSeed(42);
        loader.valTestSplit(VAL_PERCENT, TEST_PERCENT);
        loader.batch(BATCH_SIZE); 

        // Visualize a random training image from the first batch
        int randIndex = (int)(Math.random() * BATCH_SIZE);
        // Display a 28x28 image scaled up 20x for visibility
        JPlot.displayImage(
            loader.getBatches()
                .get(0) // First batch
                .get(randIndex), // Random image from batch
            20 // Scale factor
        );
        
        // Build the model
        Sequential model = new Sequential(
            "MNIST_neural_network",
            Dense(128, InputShape(FLAT_IMAGE_SIZE)),
            GELU(),

            Dense(64),
            GELU(),
            Dropout(0.1),

            Dense(NUM_CLASSES),
            Softmax()
        )
        .setDebug(DEBUG_ENABLED) 
        .summary(); // Print a model summary in the terminal


    // Try out different optimizers
        // model.compile(SGD(0.1));
        // model.compile(AdaGrad(0.01));
        // model.compile(RMSprop(0.001, 0.9, 1e-8, 0.9));
        model.compile(Adam(0.01));


    // load trained weights
        // model.loadWeights("saved_weights/MNIST_neural_network"); 

        // Train the model
        model.train(
            loader, 
            NUM_EPOCHS,
            ModelCheckpoint(
                "val_loss", 
                "saved_weights/MNIST_neural_network"
            )
        );

        // Evaluate the model
        int[] predictions = model.predict(loader.getTestImages());

        Metrics.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = Metrics.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);
    }
}