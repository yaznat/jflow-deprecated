package demos;
import jflow.data.*;
import jflow.model.Sequential;
import jflow.utils.JPlot;
import jflow.utils.Metrics;

// Static import for cleaner UI
import static jflow.model.Builder.Conv2D;
import static jflow.model.Builder.Swish;
import static jflow.model.Builder.BatchNorm;
import static jflow.model.Builder.Dropout;
import static jflow.model.Builder.MaxPool2D;
import static jflow.model.Builder.Flatten;
import static jflow.model.Builder.Dense;
import static jflow.model.Builder.Sigmoid;
import static jflow.model.Builder.Adam;
import static jflow.model.Builder.ModelCheckpoint;
/**
 * Demo to train a convolutional neural network (CNN)
 * to classify trucks vs. automobiles using the CIFAR-10 dataset.
 * 
 * Note: All library methods are documented via Javadoc. 
 * Hover in an IDE for details.
 */
public class CNN {
    // Helper method to add a block to the model
    private static void addConvBlock(Sequential model, int filters) {
        model
            .add(Conv2D(filters, 3, 1, "same_padding"))
            .add(Swish())
            .add(BatchNorm())

            .add(MaxPool2D(2, 2));

    }

    public static void main(String[] args) {
        // training constants
        final int BATCH_SIZE = 64;
        final int NUM_EPOCHS = 30;
        final double VAL_PERCENT = 0.02;
        final double TEST_PERCENT = 0.02;
        
        // Cifar10 constants
        final int COLOR_CHANNELS = 3; // RGB images
        final int IMAGE_SIZE = 32;

        // Load data
        Dataloader loader = new Dataloader();
    // Use if necessary
        // loader.setLowMemoryMode(true); 

        /*
         * Declare labels to use a train labels reference csv.
         * We only want cars and trucks.
         */ 
        String[] labelsToKeep = {"automobile","truck"};

        loader.loadFromDirectory("datasets/cifar10", labelsToKeep, 
            "datasets/CifarTrainLabels.csv", 1.0, false);


        // Prepare the transform and apply normalization
        Transform transform = new Transform()
            .normalizeTanh(); 

        loader.applyTransform(transform);

        // Prepare data for training
        loader.setSeed(42);
        loader.valTestSplit(VAL_PERCENT, TEST_PERCENT);
        loader.batch(BATCH_SIZE);

        // Visualize a random training image from the first batch
        int randIndex = (int)(Math.random() * BATCH_SIZE);
        // Display a 32x32 image scaled up 20x for visibility
        JPlot.displayImage(
            loader.getBatches()
                .get(0) // First batch
                .get(randIndex), // Random image from batch
            20 // Scale factor
        );

        // Build the model
        Sequential model = new Sequential("Cifar10_CNN");

        // Block 1
        addConvBlock(model, 32);

        // Block 2
        addConvBlock(model, 64);

        // Block 3
        addConvBlock(model, 128);

        // Flatten and Dense layers
        model
            .add(Flatten())
            .add(Dense(128))
            .add(Swish())
            .add(Dropout(0.3))

            .add(Dense(1))
            .add(Sigmoid()) // Sigmoid for binary classification

            // Pass a dummy tensor to specify input shape
            .summary(JMatrix.zeros(1, COLOR_CHANNELS, IMAGE_SIZE, IMAGE_SIZE));

    // Try out different optimizers
        // model.compile(SGD(0.01, 0.9, true));
        // model.compile(AdaGrad(0.01));
        // model.compile(RMSprop(0.001, 0.9, 1e-8, 0.9));
        model.compile(Adam(0.001));

    // Load trained weights
        // model.loadWeights("saved_weights/Cifar10 CNN Cars vs Trucks");

        // Train the model
        model.train(
            loader, 
            NUM_EPOCHS,
            ModelCheckpoint(
                "val_loss", 
                "saved_weights/Cifar10 CNN Cars vs Trucks"
            )
        );

        // Evaluate the model
        int[] predictions = model.predict(loader.getTestImages());

        Metrics.displayConfusionMatrix(predictions, loader.getTestLabels());

        double newAccuracy = Metrics.getAccuracy(predictions, loader.getTestLabels());
        System.out.println("Test accuracy:" + newAccuracy);
    }
}