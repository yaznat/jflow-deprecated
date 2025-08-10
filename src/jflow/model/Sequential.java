package jflow.model;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.IntStream;

import jflow.data.*;
import jflow.layers.Embedding;
import jflow.layers.Sigmoid;
import jflow.layers.templates.TrainableLayer;
import jflow.utils.AnsiCodes;
import jflow.utils.Callbacks;
import jflow.utils.LayerManifest;
import jflow.utils.Metrics;
import jflow.utils.Metric;



// The sequential object represents a model
public class Sequential{
    private LayerList layers = new LayerList();
    private int numClasses = -1;
    private static int sequentialCount;
    private int modelNum;
    private String name = null;
    private boolean debugMode;
    private boolean built;
    private Optimizer optimizer;
    private HashMap<String, JMatrix[]> layerGradients = new HashMap<>();
    private int[] inputShape;

    /**
     * Initializes an empty Sequential model.
     */
    public Sequential(){
        modelNum = sequentialCount++;
    }

    /**
     * Initializes an empty Sequential model.
     * @param name the name to assign to this model.
     */
    public Sequential(String name){
        this.name = name;
        modelNum = sequentialCount++;
    }

    /**
     * Initializes a Sequential model with given layers.
     * @param layers JFlow Layers to add to this model. 
     */
    public Sequential(Layer... layers){
        modelNum = sequentialCount++;
        for (Layer l : layers) {
            add(l);
        }
    }

    /**
     * Initializes a Sequential model with given layers.
     * @param name the name to assign to this Sequential model.
     * @param layers JFlow Layers to add to this model. 
     */
    public Sequential(String name, Layer... layers){
        this.name = name;
        modelNum = sequentialCount++;
        for (Layer l : layers) {
            add(l);
        }
    }

    /**
     * Returns the name of this Sequential model. If not set, defaults to
     * "sequential_X", where X is the model's initialization number.
     */
    public String name() {
        return (name == null) ? "sequential_" + modelNum : name;
    }

    /**
     * Add a layer to the model.
     * @param layer A JFlow Layer.
     */
    public Sequential add(Layer layer) {
        layers.add(layer);
        return this;
    }

    /**
     * Set the input shape of this model.
     * @param shape Either a 2D or 4D input shape, excluding the batch dimension.
     */
    public Sequential setInputShape(InputShape shape) {
        inputShape = shape.getShape();
        return this;
    }

    /**
     * Prepare the model for training.
     * @param optimizer The desired optimizer.
     */
    public Optimizer compile(Optimizer optimizer) {
        setOptimizer(optimizer);
        return optimizer;
    }

    // Initialize each trainable layer in the optimizer
    private void setOptimizer(Optimizer optimizer) {
        this.optimizer = optimizer;
        for (TrainableLayer trainable : layers.getLayersOfType(TrainableLayer.class)) {
            optimizer.initializeLayer(trainable);
            /*
             * Store references to internal gradients.
             * References always remain valid.
             */ 
            layerGradients.put(
                trainable.getName(), 
                trainable.getParameterGradients()
            );
        }
    }

     /**
     * Retrieve the parameter gradients from the model. <p>
     * For custom train steps, use: {@code optimizer.apply(model.parameterGradients())}.
     * This applies weight updates to the model layers.
     */
    public HashMap<String, JMatrix[]> parameterGradients() {
        return layerGradients;
    }

    private int countNumClasses(Dataloader loader) {
        int numImages = loader.size();
        ArrayList<Integer> uniqueLabels = new ArrayList<Integer>();
        for (int i = 0; i < numImages; i++) {
            int label = loader.get(i).getLabel();
            if (!uniqueLabels.contains(label)) {
                uniqueLabels.add(label);
            }
        }
        return uniqueLabels.size();
    }

    /**
     * Use if current train data does not contain images of all classification labels.
     * @param numClasses Set the number of classification labels.
     */
    public void setNumClasses(int numClasses) {
        this.numClasses = numClasses;
    }

    /**
     * Disable gradient storage for very large models. <p>
     * 
     * Gradient storage is enabled by default, keeping a record of the most recent forward output
     * and backward gradient for each layer. This behavior is crucial for gradient debugging, 
     * as well as allowing the user to manually access these values from a layer. In the case of 
     * very large models, it may be necessary to disable gradient storage for the sake of memory.
     */
    public Sequential disableGradientStorage() {
        for (Layer l : layers.getFlat()) {
            l.disableGradientStorage();
        }
        return this;
    }

    /**
     * Train the model.
     * @param loader                A Dataloader containing train images.
     * @param epochs                The number of epochs to train.
     */
    public void train(Dataloader loader, int epochs) {
        runTraining(loader, epochs, null);
    }

    /**
     * Train the model.
     * @param loader                A Dataloader containing train images.
     * @param epochs                The number of epochs to train.
     * @param checkpoint            A model checkpoint denoting the save path and the 
     * metric to save best.   
     */
    public void train(Dataloader loader, int epochs, ModelCheckpoint checkpoint) {
        runTraining(loader, epochs, checkpoint);
    }

    private void runTraining(Dataloader loader, int epochs,
        ModelCheckpoint checkpoint) {
        // Ensure there is an optimizer
        if (optimizer == null) {
            setOptimizer(new SGD(0.01)); // Simplest possible optimizer
        }
        // Store values to track improvement
        double prevTrainAccuracy = 0;
        double prevValAccuracy = 0;
        double prevValLoss = Double.POSITIVE_INFINITY;

        // Print training header
        Callbacks.printTrainingHeader(this);
        // Prepare validation data
        JMatrix valData = null;
        int[] valLabels = null;
        boolean useValSet = false;

        if (loader.imageBreakdown().get("val") > 0) {
            valData = loader.getValImages();
            valLabels = loader.getValLabels();
            useValSet = true;
        }
        
        int numBatches = loader.numBatches();
        int batchSize = loader.getBatch(0).size();

        int classes = (numClasses == -1) ? countNumClasses(loader) : numClasses;
        // begin training
        for (int epoch = 1; epoch <= epochs; epoch++) {
            double accuracy = 0;
            long startTime = System.nanoTime();
            double totalLoss = 0;
            for (int batch = 0; batch < numBatches; batch++) {
                JMatrix xBatch = loader.getBatchFlat(batch);
                int[] yBatch = loader.getBatchLabels(batch);

                forward(xBatch, true);

                JMatrix yTrue;
                if (layers.getLast() instanceof Sigmoid) {
                    float[] yBatchf = new float[batchSize];
                    for (int i = 0; i < batchSize; i++) {
                        yBatchf[i] = (float)yBatch[i];
                    }
                    yTrue = JMatrix.wrap(yBatchf, batchSize, 1, 1, 1);
                    
                } else {
                    yTrue = oneHotEncode(yBatch, classes, false);
                }

                backward(yTrue);

                // Apply updates
                optimizer.apply(layerGradients);

                JMatrix output = layers.getLast().getOutput();

                int[] predictions = getPredictions(output);

                accuracy += Metrics.getAccuracy(predictions, yBatch);
                totalLoss += crossEntropyLoss(output, yBatch);

                long batchTime = System.nanoTime();
                long timeSinceStart = batchTime - startTime;

                LinkedHashMap<String, Double> lossReport = new LinkedHashMap<>();
                lossReport.put("Loss", totalLoss / (batch + 1));

                Callbacks.printProgressCallback("Epoch", epoch, epochs, "Batch", batch + 1, numBatches,
                        timeSinceStart, lossReport);
                if (debugMode) {
                    System.out.println(); // No carriage return
                }
            }
            Double trainLoss = totalLoss / numBatches;

            // Gather metric values to report for this epoch
            List<Metric> metricReport = new ArrayList<>();

            // Add train accuracy
            double trainAccuracy = accuracy / numBatches;
            metricReport.add(new Metric(
                "Training Accuracy", 
                trainAccuracy, 
                true, 
                trainAccuracy > prevTrainAccuracy 
                )
            );
            prevTrainAccuracy = trainAccuracy;

            double valLoss = Double.POSITIVE_INFINITY;
            double valAccuracy = 0;
            if (useValSet) {
                // Test on the validation set
                int[] valPredictions = predict(valData);

                // Add validation accuracy
                valAccuracy = Metrics.getAccuracy(valPredictions, valLabels);

                metricReport.add(new Metric(
                    "Validation Accuracy", 
                    valAccuracy, 
                    true, 
                    valAccuracy > prevValAccuracy 
                    )
                );
                prevValAccuracy = valAccuracy;

                // Add validation loss
                valLoss = crossEntropyLoss(layers.getLast().getOutput(), valLabels);

                metricReport.add(new Metric(
                    "Validation Loss", 
                    valLoss, 
                    false, 
                    valLoss < prevValLoss 
                    )
                );
                prevValLoss = valLoss;
            }
            Callbacks.printMetricCallback(metricReport);

            if (checkpoint != null) {
                checkpoint.updateAndPrintCallback(
                    trainAccuracy, 
                    trainLoss, 
                    valAccuracy, 
                    valLoss
                );

                if (checkpoint.improved()) {
                    internalSaveWeights(checkpoint.getSavePath(), false);
                }
            }
            System.out.println();

            
        }
    }

    /**
     * Predict class labels on batched image data in a JMatrix.
     * @param images                    a JMatrix of images in the shape (N, channels, height, width).
     * @return                      predicted class labels in the range [0, numClasses].
     */
    public int[] predict(JMatrix images) {
        // Forward pass
        JMatrix output = forward(images, false);

        // Get predictions
        return getPredictions(output);
        
    }

    // Internal helper to convert output to predictions
    private int[] getPredictions(JMatrix output) {
        if (layers.getLast() instanceof Sigmoid) {
            int batchSize = output.shape(0);
            int[] predictions = new int[batchSize];
            for (int i = 0; i < batchSize; i++) {
                predictions[i] = (output.get(i) >= 0.5) ? 1 : 0;
            }
            return predictions;
        }
        return output.argmax(1);
    }

    /**
     * When enabled, prints a detailed, ANSI-styled debug output 
     * for each layer during calls to forward() and backward()
     * @param enabled               Set debug mode to on or off.
     */
    public Sequential setDebug(boolean enabled) {
        debugMode = enabled;
        for (Layer l : layers.getFlat()) {
            if (enabled) {
                l.enableDebugForThisLayer();
            } else {
                l.disableDebugForThisLayer();
            }
        }
        return this;
    }
    
    // One hot encode labels
    private JMatrix oneHotEncode(int[] labels, int numClasses,
                                 boolean transpose) throws IllegalArgumentException {
        JMatrix oneHot = JMatrix.zeros(labels.length, numClasses, 1, 1);
        float[] oneHotMatrix = oneHot.unwrap();
        for (int x = 0; x < labels.length; x++) {
            oneHotMatrix[x * numClasses + labels[x]] = 1.0f;
        }
        if (transpose) {
            oneHot = oneHot.T();
        }
        return oneHot;
    }

    /**
     * Force this model to build. Calling this method is not typically required. Build is automatically handled by calling:
     * {@code model.summary(...)}, {@code model.forward(...)}, or {@code model.loadWeights(...)}.
     */
    public Sequential build() {
        if (!built) {
            if (inputShape != null) {
                forward(JMatrix.zeros(inputShape), false);
            } else {
                int[] layerInputShape = layers.getFirst().getInputShape();
                if (layerInputShape != null) {
                    forward(JMatrix.zeros(layerInputShape), false);
                } else {
                    // Embedding can usually be built with (batchSize = 1, seqLen = 1)
                    List<TrainableLayer> trainables = layers.getLayersOfType(TrainableLayer.class);
                    if (trainables.getFirst() instanceof Embedding) {
                        try {
                            forward(JMatrix.zeros(1, 1, 1, 1), false);
                        } catch (Exception e) {
                            e.printStackTrace();
                            throw new IllegalStateException(
                                "Unusual model architecture detected. " + 
                                "Input shape must be provided to properly build the model. " +
                                "Either pass jflow.model.Builder.InputShape as an argument in the Embedding constructor, "
                                + "or call model.setInputShape(...)." + 
                                "\nNote: if input shape is set and correct, the error may come from your FunctionalLayer code. " + 
                                "See the stack trace above."
                            );
                        }
                    }
                }
            }
            built = true;
        }
        return this;
    }

    /**
     * Perform forward propagation.
     * @param input                Input data wrapped in a JMatrix.
     * @param training             Indicate whether for training or inference.
     * @return                     Returns the forward output of the last layer of the model.
     */
    public JMatrix forward(JMatrix input, boolean training) {
        if (debugMode && training) {
            System.out.println(
                AnsiCodes.BLUE + "============ " + 
                "Forward Debug: " + AnsiCodes.BOLD + name() + AnsiCodes.RESET + 
                AnsiCodes.BLUE + " ============" + AnsiCodes.RESET
            );
            Callbacks.printStats("", input.label("input"));

        }
        JMatrix output = input;
        // Only call the outermost layers externally
        for (Layer layer : layers.getLevel(0)) {
            output = layer.forward(output, training);

            if (debugMode) {
                layer.printForwardDebug();
            }
        }

        built = true; // If forward pass was successful
        
        return output;
    }

    /**
     * Perform backward propagation.
     * @param gradient               Gradient wrapped in a JMatrix.
     * @param learningRate         The desired learning rate for updating parameters.
     * @return                     Returns the gradient, dInput, of the first layer of the model.
     */
    public JMatrix backward(JMatrix gradient) {
        if (debugMode) {
            System.out.println(
                AnsiCodes.BLUE + "============ " + 
                "Backward Debug: " + AnsiCodes.BOLD + name() + AnsiCodes.RESET + 
                AnsiCodes.BLUE + " ============" + AnsiCodes.RESET
            );
            Callbacks.printStats("", gradient.label("gradient"));
        }

        // Only call the outermost layers externally
        JMatrix dInput = gradient;
        for (Layer layer : layers.getLevel(0).reversed()) {
            dInput = layer.backward(dInput);

            if (debugMode) {
                layer.printBackwardDebug();
            }
        }

        return dInput;
    }

    // Calculate loss per batch
    private double crossEntropyLoss(JMatrix output, int[] labels) {
        double epsilon = 1e-12;
        int batchSize = labels.length;
        int numClasses = output.shape(1);
        double totalLoss = 0;

        if (layers.getLast() instanceof Sigmoid) {
            // b.c.e. for sigmoid activation
            for (int i = 0; i < batchSize; i++) {
                int label = labels[i];
                double predictedProb = output.get(i);
                // b.c.e.
                totalLoss += -label * Math.log(predictedProb + epsilon) - 
                            (1 - label) * Math.log(1 - predictedProb + epsilon);
            }
        } else {
            for (int i = 0; i < batchSize; i++) {
                int label = labels[i];
                int index = i * numClasses + label;
                double predictedProb = output.get(index);
                totalLoss += -Math.log(predictedProb + epsilon);
            }
        }
    
        return totalLoss / batchSize;
    }

    /**
     * Save weights to binary files in a directory.
     * @param path               The name of the directory to store files in.
     */
    public void saveWeights(String path) {
        internalSaveWeights(path, true);
    }
    public void internalSaveWeights(String path, boolean printReport) {
        List<TrainableLayer> trainableLayers = layers.getLayersOfType(TrainableLayer.class);
        // Save trainable layer weights
        IntStream.range(0, trainableLayers.size())
            .parallel()
            .forEach(i -> {
                TrainableLayer trainable = trainableLayers.get(i);
                for (JMatrix weight : trainable.getParameters()) {
                    String filePath = path + "/" + trainable.getName() + "_" + weight.label() + ".bin";
                    saveWeightToBinary(filePath, weight);
                }
                
            });
    
        // Save optimizer time steps
        if (optimizer != null) {
            if (optimizer instanceof Adam adam) {
                String timestepPath = path + "/" + optimizer.getName() + "/timesteps.bin";
                try {
                    Path dir = Paths.get(path + "/" + optimizer.getName());
                    Files.createDirectories(dir);
            
                    try (DataOutputStream dos = new DataOutputStream(new FileOutputStream(timestepPath))) {
                        dos.writeLong(adam.getTimeSteps()); // Write 8-byte long
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
    
            // Save optimizer moments
            JMatrix[] moments = optimizer.getSerializable();
            IntStream.range(0, moments.length)
                .parallel()
                .forEach(i -> {
                    JMatrix moment = moments[i];
                    String filePath = path + "/" + optimizer.getName() + "/" + moment.label() + ".bin";
                    saveWeightToBinary(filePath, moment);
                });
        }
    
        if (printReport) {
            System.out.println("Weights saved to " + path);
        }
    }

    // Helper method to write weight values to binary
    private void saveWeightToBinary(String filePath, JMatrix weight) {
        try {
            Path dir = Paths.get(filePath).getParent();
            if (dir != null) Files.createDirectories(dir);

            try (DataOutputStream dos = new DataOutputStream(
                    new BufferedOutputStream(
                        new FileOutputStream(filePath)))) {
                for (int i = 0; i < weight.size(); i++) {
                    dos.writeFloat(weight.get(i)); // Write 4 bytes per float
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /**
     * Load weights from binary files in a directory.
     * @param path               The location of the directory to load files from.
     */
    public void loadWeights(String path) {
        build();
        List<TrainableLayer> trainableLayers = layers.getLayersOfType(TrainableLayer.class);
        // Load all trainable layer weights
        IntStream.range(0, trainableLayers.size())
            .parallel()
            .forEach(i -> {
                TrainableLayer trainable = trainableLayers.get(i);
              
                JMatrix[] weights = trainable.getParameters();
                for (JMatrix weight : weights) {
                    String filePath = path + "/" + trainable.getName() + "_" + weight.label() + ".bin";
                    loadWeightFromBinary(filePath, weight);
                }
            });

        if (optimizer != null) {
            // Load timestep
            if (optimizer instanceof Adam adam) {
                String timestepPath = path + "/" + optimizer.getName() + "/timesteps.bin";
                try (DataInputStream dis = new DataInputStream(new FileInputStream(timestepPath))) {
                    adam.setTimeSteps(dis.readLong());
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            // Load optimizer moments
            JMatrix[] moments = optimizer.getSerializable();
            IntStream.range(0, moments.length)
                .parallel()
                .forEach(i -> {
                    JMatrix moment = moments[i];
                    String filePath = path + "/" + optimizer.getName() + "/" + moment.label() + ".bin";
                    loadWeightFromBinary(filePath, moment);
                });
        }
    }

    // Helper method to read a binary file into a JMatrix
    private void loadWeightFromBinary(String filePath, JMatrix weight) {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(filePath)))) {
            for (int i = 0; i < weight.size(); i++) {
                weight.set(i, dis.readFloat());
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Print a model summary in the terminal.
     */
    public Sequential summary() {
        build();
        // Try to dynamically infer the input tensor
        if (inputShape != null) {
            summary(JMatrix.zeros(inputShape));
        } else {
            int[] layerInputShape = layers.getFirst().getInputShape();
            if (layerInputShape != null) {
                summary(JMatrix.zeros(layerInputShape));
            } else {
                throw new IllegalStateException(
                "Model input shape must be set for dynamic summary. " + 
                "Call model.setInputShape(...) or call model.summary(JMatrix inputTensor) for a specified summary."
            );
            }  
        }
        return this;
    }

    private String formatLayerColor(String simpleClassName) {
        if (LayerManifest.isSupported(simpleClassName)) {
            return AnsiCodes.TEAL;
        }
        switch(simpleClassName) {
            case "type":
                return AnsiCodes.BOLD + AnsiCodes.TEAL;
            case "InputLayer":
                return AnsiCodes.ITALIC + AnsiCodes.PURPLE;
            default:
                return AnsiCodes.GRAY; // FunctionalLayers
        }

    }

    /**
     * Print a model summary in the terminal.
     * @param inputTensor Specify an input tensor to visualize how it passes through the model.
     */
    public Sequential summary(JMatrix inputTensor) {
        // Run a dummy pass to build output shapes
        forward(inputTensor, true);

        // Only include the outermost level in the summary
        List<Layer> summaryLayers = layers.getLevel(0);

        // Count the total number of layers that aren't FunctionalLayers
        int numLayers = summaryLayers.size();

        // Sum trainable paramters
        int trainableParameters = 0;
        for (Layer l : summaryLayers) {
            trainableParameters += l.numTrainableParameters();
        }

        // Declare the size of each column
        final int minSpacesType = 30;
        final int minSpacesShape = 15;
        final int minSpacesParam = 15;

        int spacesType = minSpacesType;
        int spacesShape = minSpacesShape;
        int spacesParam = minSpacesParam;

        int requiredGap = 5;

        for (Layer l : summaryLayers) {
            int typeLength = l.getName().length() + l.getType().length() + 3;
            int shapeLength = l.getOutput().simpleShapeAsString().length();
            int paramLength = String.valueOf(l.numTrainableParameters()).length();

            spacesType = Math.max(spacesType, typeLength + requiredGap);
            spacesShape = Math.max(spacesShape, shapeLength + requiredGap);
            spacesParam = Math.max(spacesParam, paramLength + requiredGap);
        }

        // Display the layer names and types
        String[][] layerTypes = new String[numLayers + 2][2];
        layerTypes[0][0] = "Layer";
        layerTypes[0][1] = "type";
        layerTypes[1][0] = "input";
        layerTypes[1][1] = "InputLayer";

        int targetIndex = 2; // Start after header
        for (Layer layer : summaryLayers) {
            String type = layer.getClass().getSimpleName();
            String name = layer.getName();
            layerTypes[targetIndex][0] = name;
            layerTypes[targetIndex][1] = type;
            targetIndex++;
        }

        String[] shapes = new String[numLayers + 2];
        shapes[0] = "Output Shape";
        shapes[1] = inputTensor.simpleShapeAsString();

        targetIndex = 2; // Start after header
        for (Layer layer : summaryLayers) {
            shapes[targetIndex++] = layer.getOutput().simpleShapeAsString();
        }
        String[] params = new String[numLayers + 2];
        params[0] = "Param #";
        params[1] = "--";
        
        targetIndex = 2; // Start after header
        for (Layer layer : summaryLayers) {
            String paramCountAsString = String.valueOf(layer.numTrainableParameters());
            if (paramCountAsString.length() > 6) {
                params[targetIndex++] = NumberFormat.getIntegerInstance().format(layer.numTrainableParameters());
            } else {
                params[targetIndex++] = paramCountAsString;
            }
        }

        String title = 
            AnsiCodes.BOLD + AnsiCodes.WHITE + 
            " Model Summary" + " (" + name() + ")"
            + AnsiCodes.RESET;

        // Print title and summary top shell
        System.out.println(
            "\n" + title + "\n" + AnsiCodes.BLUE + 
            "╭" + "─".repeat(spacesType) + "┬" + 
            "─".repeat(spacesShape) + "┬" + 
            "─".repeat(spacesParam) + "╮" + 
            AnsiCodes.RESET
        
        );

        final String cellWall = AnsiCodes.BLUE + "│ " + AnsiCodes.RESET;
        // Print the body of the summary
        for (int line = 0; line < layerTypes.length; line++) {
            int numSpaces = spacesType - (layerTypes[line][0].length() + 
                layerTypes[line][1].length() + 4);

            System.out.print(cellWall);

            // Print a "Layer (type)" item
            System.out.print(
                formatLayerColor(layerTypes[line][1]) + layerTypes[line][0] + 
                AnsiCodes.WHITE + " (" + layerTypes[line][1] 
                + ")" + AnsiCodes.RESET + " ".repeat(numSpaces)
            );

            System.out.print(cellWall);

            // Print an "Output Shape" item
            numSpaces = spacesShape - shapes[line].length() - 1;
            if (line == 0) {
                System.out.print(AnsiCodes.BOLD + AnsiCodes.WHITE + shapes[0]);
            } else {
                System.out.print(
                    AnsiCodes.WHITE + "("
                );
                String[] dims = shapes[line].replace("(", "").replace(")", "").split(",");
                for (int i = 0; i < dims.length; i++) {
                    System.out.print(AnsiCodes.DARK_ORANGE + dims[i]);
                    if (i == 0 || !(i == dims.length - 1)) {
                        System.out.print(AnsiCodes.WHITE + ",");
                    }
                }
                System.out.print(AnsiCodes.WHITE + ")");
            }
            System.out.print(" ".repeat(numSpaces));
            System.out.print(cellWall);

            // Print a "Param #" item, right-centered
            if (line == 0) {
                System.out.print(AnsiCodes.BOLD);
            }
            numSpaces = spacesParam - params[line].length() - 1;
            System.out.print(" ".repeat(numSpaces));
            
            System.out.print(AnsiCodes.WHITE + params[line]);
            
            System.out.println(cellWall + AnsiCodes.BLUE);

            if (line == layerTypes.length - 1) {
                System.out.print("╰");
            } else {
                System.out.print("├");
            }
            System.out.print("─".repeat(spacesType));

            if (line == layerTypes.length - 1) {
                System.out.print("┴");
            } else {
                System.out.print("┼");
            }
            System.out.print("─".repeat(spacesShape));

            if (line == layerTypes.length - 1) {
                System.out.print("┴");
            } else {
                System.out.print("┼");
            }
            System.out.print("─".repeat(spacesParam));
            
            if (line == layerTypes.length - 1) {
                System.out.print("╯");
            }
            else {
                System.out.print("┤");
            }
            System.out.println();
        }
        // Format the parameter count with commas for readability
        String formatted = NumberFormat.getIntegerInstance().format(trainableParameters);

        // Calculate parameter size in MB
        double sizeInMB = trainableParameters / 250000.0;

        System.out.println(
            AnsiCodes.BOLD + AnsiCodes.DARK_ORANGE + 
            "Total params: " + AnsiCodes.WHITE +
            formatted + " (" + String.format("%.1f", sizeInMB) + " MB)"
            + AnsiCodes.RESET
        );

        return this;
    }
}
