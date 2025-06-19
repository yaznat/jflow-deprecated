package jflow.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.function.Function;
import java.util.stream.IntStream;

public class Dataloader {
    private ArrayList<Image> images = new ArrayList<>();

    private Random random = new Random(0);

    private ArrayList<Image> trainImages = new ArrayList<>(); 
    private ArrayList<Image> testImages = new ArrayList<>(); 
    private ArrayList<Image> valImages = new ArrayList<>();

    private int batchSize = -1;
    private int numBatches;

    private boolean lowMemoryMode = false;


    /**
     * Initializes an empty Dataloader.
     */
    public Dataloader () {}

    /**
     * When low memory mode is enabled, 
     * images are kept in the unloaded
     * state. This allows for an 
     * unlimited dataset size during training,
     * with a small increase in duration.
     * @param enabled               Set the state of low memory mode.
     */
    public void setLowMemoryMode(boolean enabled) {
        lowMemoryMode = enabled;
    }

    /**
     * @return True if low memory mode is enabled.
     */
    public boolean isLowMemoryModeEnabled() {
        return lowMemoryMode;
    }

    /**
     * Applies a constructed transform to all images in the Dataloader.
     * @param transform                 A transform containing image functions.
     */
    public void applyTransform(Transform transform) {
        for (Image image : images) {
            for (Function<JMatrix, JMatrix> func : transform.getTransforms()) {
                image.addTransform(func);
            }
        }
    }

    /**
      * Loads a portion of image files from a directory and assigns a label to them.
      *
      * @param directory           The directory where image files are stored.
      * @param label               The label to assign to all loaded images.
      * @param percentOfDirectory  The percentage of the directory to load (0.0 to 1.0).
      * @param grayscale           Whether the images are grayscale. Required due to a known detection bug.
      */
    public void loadFromDirectory(String directory, int label, double percentOfDirectory, boolean grayscale) {
        File dir = new File(directory);

        File[] files = dir.listFiles();

        int numImages = (int)(files.length * percentOfDirectory);

        for (int i = 0; i < numImages; i++) {
            if (files[i].getAbsolutePath().endsWith(".png") || 
                    files[i].getAbsolutePath().endsWith(".jpg"))
                images.add(new Image(files[i].getAbsolutePath(), label, grayscale, lowMemoryMode));
        }
    }

    /**
      * Loads a portion of image files from a directory, assigns a label to them, and performs resize.
      *
      * @param directory           The directory where image files are stored.
      * @param label               The label to assign to all loaded images.
      * @param percentOfDirectory  The percentage of the directory to load (0.0 to 1.0).
      * @param grayscale           Whether the images are grayscale. Required due to a known detection bug.
      * @param resize              Resize images to [height, width]
      */
    public void loadFromDirectory(String directory, int label, double percentOfDirectory, boolean grayscale, int[] resize) {
        File dir = new File(directory);

        File[] files = dir.listFiles();

        int numImages = (int)(files.length * percentOfDirectory);

        Transform transform = new Transform();
        transform.resize(resize[0], resize[1]);

        Function<JMatrix, JMatrix> resizeFunc = transform.getTransforms().get(0);

        for (int i = 0; i < numImages; i++) {
            if (files[i].getAbsolutePath().endsWith(".png") || 
                    files[i].getAbsolutePath().endsWith(".jpg")) {
                images.add(new Image(files[i].getAbsolutePath(), label, grayscale, lowMemoryMode));
                images.getLast().addTransform(resizeFunc);
            }
        }
    }

    /**
      * Loads a portion of image files from a directory and finds labels in TrainLabels.csv
      *
      * @param directory           The directory where image files are stored.
      * @param labelsInOrder       The desired classes of images to keep, in order.
      * @param pathToLabelCSV      The location of TrainLabels.csv or similar.
      * @param percentOfDirectory  The percentage of the directory to load (0.0 to 1.0).
      * @param grayscale           Whether the images are grayscale. Required due to a known detection bug.
      */
    public void loadFromDirectory(String directory, String[] labelsInOrder, String pathToLabelCSV, double percentOfDirectory, boolean grayscale) {
        File dir = new File(directory);

        File[] files = dir.listFiles();

        Arrays.sort(files, Comparator.comparingInt(file -> extractNumber(file.getName())));
        int numImages = (int)(files.length * percentOfDirectory);

        try(BufferedReader br = new BufferedReader(new FileReader(pathToLabelCSV))) {
            String line;
            int index = 0;
            while((line = br.readLine()) != null && index < numImages){
                String labelName = line.split(",")[1];
                int label = indexOf(labelsInOrder, labelName);
                if ((files[index].getAbsolutePath().endsWith(".png") || 
                    files[index].getAbsolutePath().endsWith(".jpg")) && 
                    label != -1)
                    images.add(new Image(files[index].getAbsolutePath(), label, grayscale, lowMemoryMode));
                index++;
            }
        } catch (Exception e) {
            System.err.println(e);
        }
    }


    private int extractNumber(String filename) {
        try {
            return Integer.parseInt(filename.replaceAll("[^0-9]", ""));
        } catch (NumberFormatException e) {
            return Integer.MAX_VALUE; // Push non-numeric names to the end
        }
    }
    

    /**
      * Loads flattened grayscale images from a .csv or .txt file.
      *
      * @param path                The path to the file.
      * @param areLabelsFirstItem  Whether labels are the first item of each row.
      * @param percentOfFile       The percentage of the file to load (0.0 to 1.0).
      */
    public void loadFromCSV(String path, boolean areLabelsFirstItem, double percentOfFile) {
        ArrayList<Image> loadedImages = new ArrayList<Image>();
        try(BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line; String[] split;
            int label = 0;
            while((line = br.readLine()) != null){
                split = line.split(",");
                float[] image;
                int index = 0;
                if (areLabelsFirstItem) {
                    label = Integer.valueOf(split[0]);
                    index = 1;
                    image = new float[split.length - 1];
                } else {
                    image = new float[split.length];
                }
                for (int i = index; i < split.length; i++) {
                    image[i - index] = Float.parseFloat(split[i]);
                }
                
                loadedImages.add(new Image(image, label));
            }
        } catch (Exception e) {
            System.err.println(e);
        }
        int imagesToKeep = (int)(percentOfFile * loadedImages.size());

        for (int i = 0; i< imagesToKeep; i++) {
            images.add(loadedImages.get(i));
        }
    }
    /**
      * Add a single image to the Dataloader.
      *
      * @param array               An array representing the image in the format (channels, height, width)
      * @param label               The class label of the image.
      */
    public void addMatrixAsImage(JMatrix matrix, int label) {
        images.add(new Image(matrix, label));
    }
    /**
      * Removes all images from the Dataloader.
      */
    public void clear() {
        images = new ArrayList<Image>();
        trainImages = new ArrayList<Image>();
        testImages = new ArrayList<Image>();
    }

    /**
      * Sets the seed of the Dataloader for reproducability.
      * @param seed The desired seed for random operations.
      */
    public void setSeed(long seed) {
        random.setSeed(seed);
    }

    /**
      * Separates train images into batches.
      * @param batchSize The desired batch size.
      */
    public void batch(int batchSize) {
        this.batchSize = batchSize;
        this.numBatches = trainImages.size()/batchSize;
    }

    /**
      * Shuffles images in the Dataloader. Not effective after train-test split.
      */
    public void shuffle() {
        for (int i = 0; i < images.size(); i++) {
            int randIndex = random.nextInt(images.size());

            Image temp = images.get(i);
            images.set(i, images.get(randIndex));

            images.set(randIndex, temp);
        }
    }

    /**
     * Keep only images in the Dataloader with the provided labels.
     * @param labels                The class labels to keep.
     */
    public void filterByLabel(int[] labels) {
        for (int i = 0; i < images.size(); i++) {
            Image image = images.get(i);
            boolean match = false;
            for (int l : labels) {
                if (image.getLabel() == l) {
                    match = true;
                    break;
                }
            }
            if (!match) {
                images.remove(i);
                trainImages.remove(image);
                testImages.remove(image);
                i--;
            }
        }
    }

    /**
      * Get an image from the dataloader.
      * @param index The index of the image.
      */
    public Image get(int index) {
        return images.get(index);
    }

    /**
      * The number of images in the Dataloader.
      */
    public int size() {
        return images.size();
    }

    private int numberOfBatches() {
        if (trainImages.isEmpty()) {
            this.numBatches = images.size() / batchSize;
            return images.size() / batchSize;
        }
        this.numBatches = trainImages.size() / batchSize;
        return trainImages.size() / batchSize;
    }


    /**
      * The number of batches among train images.
      */
    public int numBatches() {
        return numBatches;
    }

    /**
      * Get a batch of train images.
      * @param index                        The index of the batch.
      * @return                             A list containing all images in the batch.
      * @throws IndexOutOfBoundsException   If the batch exceeds the length of training data.
      */
    public List<Image> getBatch(int index) {
        int beginIndex = batchSize * index;
        int endIndex = beginIndex + batchSize;
        ArrayList<Image> arrayToUse;
        if (trainImages.isEmpty()) {
            arrayToUse = images;
        } else {
            arrayToUse = trainImages;
        }
        if (endIndex <= arrayToUse.size()) {
            return arrayToUse.subList(beginIndex, endIndex);
        }
        throw new IndexOutOfBoundsException(
            "Invalid batch index for batch size " + batchSize
            + " and train data size: " + arrayToUse.size() + " images.");
    }

    /**
    * Get a batch of images flattened into a JMatrix.
    * @param index             The index of the batch among training batches.
    */
    public JMatrix getBatchFlat(int index) {
        List<Image> batch = getBatch(index);
        int channels = batch.get(0).numChannels();
        int height = batch.get(0).getHeight();
        int width = batch.get(0).getWidth();

        JMatrix flattenedBatch = JMatrix.zeros(batchSize, channels, height, width);

        IntStream.range(0, batchSize).parallel().forEach(i -> {
            JMatrix image = batch.get(i).getData();
            flattenedBatch.arrayCopyBatch(i, image);
        });
        return flattenedBatch;
    }


    /**
    * Get the labels of a batch.
    * @param index             The index of the batch among training batches.
    */
    public int[] getBatchLabels(int index) {
        List<Image> batch = getBatch(index);

        int[] labels = new int[batchSize];

        for (int i = 0; i < batchSize; i++) {
            labels[i] = batch.get(i).getLabel();
        }

        return labels;
    }

     /**
      * Get all of the training batches.
      * @returns A list of batches represented as lists of images.
      */
    public List<List<Image>> getBatches() {
        if (trainImages.isEmpty()) {
            trainImages = images;
        }
        List<List<Image>> batches = new ArrayList<>();
        for (int i = 0; i < numberOfBatches(); i++) {
            int beginIndex = batchSize * i;
            int endIndex = beginIndex + batchSize;
            batches.add(trainImages.subList(beginIndex, endIndex));
        }
        return batches;
    }

     /**
      * Shuffle and split images in the Dataloader into train and test.
      * @param percentTrain The percentage of images in the train set (0.0 to 1.0).
      */
    public void trainTestSplit(double percentTrain) {
        Collections.shuffle(images, random);
    
        int numTrainImages = (int) (images.size() * percentTrain);
    
        trainImages = new ArrayList<>(images.subList(0, numTrainImages));
        testImages = new ArrayList<>(images.subList(numTrainImages, images.size()));
    
        System.out.println("Train images: " + trainImages.size());
        System.out.println("Test images: " + testImages.size());
    }

     /**
      * Shuffle and split images in the Dataloader into train, val, and test.
      * @param percentVal                       The percentage of images in the validation set (0.0 to 1.0).
      * @param percentTest                      The percentage of images in the test set (0.0 to 1.0).
      */
      public void valTestSplit(double percentVal, double percentTest) {
        Collections.shuffle(images, random);

        int numValImages = (int) (images.size() * percentVal);
        int numTestImages = (int) (images.size() * percentTest);
        int numTrainImages = images.size() - numValImages - numTestImages;
    
        trainImages = new ArrayList<>(images.subList(0, numTrainImages));
        valImages = new ArrayList<>(images.subList(numTrainImages, numTrainImages + numValImages));
        testImages = new ArrayList<>(images.subList(numTrainImages + numValImages, images.size()));
    
        System.out.println("Train images: " + trainImages.size());
        System.out.println("Val images: " + valImages.size());
        System.out.println("Test images: " + testImages.size());
    }

    /**
      * Apply augmentation functions to train images only.
      * @param augmentations         A Transform with stored augmentation functions.
      */
    public void applyAugmentations(Transform augmentations) {
        ArrayList<Image> arrayToUse;
        if (trainImages.isEmpty()) {
            arrayToUse = images;
        } else {
            arrayToUse = trainImages;
        }
        int numImages = arrayToUse.size();
        for (int i = 0; i < numImages; i++) {
            Image augmented = arrayToUse.get(i);
            for  (Function<JMatrix, JMatrix> 
                function : augmentations.getTransforms()) {
                
                augmented.addTransform(function);
            }
        }
    }

    /**
      * Get test images in the shape (N, channels, height, width).
      * @return                                         a JMatrix with shape (N, channels, height, width).
      * @throws NullPointerException                    if the test dataset is never set.
      */
    public JMatrix getTestImages() {
        if (testImages.isEmpty()) {
            throw new NullPointerException("Test dataset never set.");
        }
        int numImages = testImages.size();
        int channels = testImages.get(0).numChannels();
        int height = testImages.get(0).getHeight();
        int width = testImages.get(0).getWidth();
        int imageSize = channels * height * width;

        // Create a JMatrix with image dimensions
        JMatrix imageBatch = JMatrix.zeros(numImages, channels, height, width);

        // Copy data into the JMatrix
        for (int i = 0; i < numImages; i++) {
            float[] image = testImages.get(i).getData().unwrap();
            for (int j = 0; j < imageSize; j++) {
                imageBatch.set(i * imageSize + j, image[j]);
            }
        }

        return imageBatch;
    }

    /**
      * Get validation images in the shape (N, channels, height, width).
      * @return                                         a JMatrix with shape (N, channels, height, width).
      * @throws NullPointerException                    if the validation dataset is never set.
      */
      public JMatrix getValImages() {
        if (valImages.isEmpty()) {
            throw new NullPointerException("Validation dataset never set.");
        }
        int numImages = valImages.size();
        int channels = valImages.get(0).numChannels();
        int height = valImages.get(0).getHeight();
        int width = valImages.get(0).getWidth();
        int imageSize = channels * height * width;

        // Create a JMatrix with image dimensions
        JMatrix imageBatch = JMatrix.zeros(numImages, channels, height, width);

        // Copy data into the JMatrix
        for (int i = 0; i < numImages; i++) {
            float[] image = valImages.get(i).getData().unwrap();
            for (int j = 0; j < imageSize; j++) {
                imageBatch.set(i * imageSize + j, image[j]);
            }
        }
        return imageBatch;
    }

    /**
      * Get test labels in order as an array.
      */
    public int[] getTestLabels() {
        if (testImages.isEmpty()) {
            throw new NullPointerException("Test dataset never set");
        }
        int numImages = testImages.size();
        int[] labels = new int[numImages];

        for (int i = 0; i < numImages; i++) {
            labels[i] = testImages.get(i).getLabel();
        }

        return labels;
    }

    /**
      * Get validation labels in order as an array.
      */
      public int[] getValLabels() {
        if (valImages.isEmpty()) {
            throw new NullPointerException("Validation dataset never set");
        }
        int numImages = valImages.size();
        int[] labels = new int[numImages];

        for (int i = 0; i < numImages; i++) {
            labels[i] = valImages.get(i).getLabel();
        }

        return labels;
    }

    /**
     * Get a report of the counts of images among train, validation, and test.
     * @return a Hashmap containing set names and their counts of images.
     */
    public HashMap<String, Integer> imageBreakdown() {
        HashMap<String, Integer> breakdown = new HashMap<>();

        breakdown.put("train", trainImages.size());
        breakdown.put("val", valImages.size());
        breakdown.put("test", testImages.size());

        return breakdown;
    }

    private int indexOf(String[] arr, String target) {
        for (int i = 0; i < arr.length; i++) {
            if (arr[i].equals(target)) {
                return i;
            }
        }
        return -1;
    }
}