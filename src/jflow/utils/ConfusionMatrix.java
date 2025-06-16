package jflow.utils;

import java.awt.Color;
import java.awt.Graphics;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ForkJoinPool;

import javax.swing.JFrame;
import javax.swing.JPanel;


class ConfusionMatrix extends JPanel{
    private JFrame frame;
    private int numClasses;
    private int[] predictions, labels;

    // This contstuctor displays a confusion matrix in a new JFrame
    ConfusionMatrix(int[] predictions, int[] labels){
        this.predictions = predictions;
        this.labels = labels;
        this.numClasses = max(labels) + 1;
        this.frame = new JFrame("Confusion Matrix");
        frame.setBounds(0, 0, 40 * numClasses + 80, 40 * numClasses + 25 + 40);
        frame.add(this);
        frame.setResizable(false);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        repaint();
        frame.setVisible(true);
    }
    protected void paintComponent(Graphics g) {
        // initialize matrix
        double[][] labelsGuessed = new double[numClasses][numClasses];
        int[] correctPerLabel = new int[numClasses];
        int[] countPerLabel = new int[numClasses];
        // for each label in matrix
        for (int label = 0; label < numClasses; label++) {
            // find yTrue that equals label
            for (int i = 0; i < predictions.length; i++) {
                if (labels[i] == label) {
                    // note what prediction corresponds to the yTrue value
                    int predicted = predictions[i];
                    labelsGuessed[label][predicted]++;
                    if (predicted == labels[i]) {
                        correctPerLabel[label]++;
                    }
                    countPerLabel[label]++;
                }
            }
        }
        double[] percentPerLabel = new double[numClasses];
        for (int label = 0; label < numClasses; label++) {
            percentPerLabel[label] = 1.0 * correctPerLabel[label] / countPerLabel[label];
        }
        try {
            // Rotate the matrix for standard configuration
            labelsGuessed = rotate90(rotate90(rotate90(labelsGuessed)));
        } catch (IllegalArgumentException e) {}
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                int red = 0; int green = 0; int blue = 0;
                if (i == j) {
                    if (percentPerLabel[j] > 0.2) {
                         red = 0;
                        green = Math.max(60, Math.min(40 * (int)(4 * (Math.pow(percentPerLabel[j], 2))), 255));;
                        blue = 0; 
                    } else {
                        red = 60;
                        green = 60;
                        blue = 90;
                    }
                } else {
                    red = 60;
                    green = 0;
                    blue = 90;
                }
                g.setColor(new Color(red, green, blue));
                g.fillRect((j + 1) * 40, i * 40, 40, 40);
                if (red > 200) {
                    g.setColor(Color.BLACK);
                } else {
                    g.setColor(Color.WHITE);
                }
                if (labelsGuessed[i][j] < 10) {
                    g.drawString(String.valueOf((int)labelsGuessed[i][j]), (j + 1) * (40) + 15, i * 40 + 23);
                } else if (labelsGuessed[i][j] < 99) {
                    g.drawString(String.valueOf((int)labelsGuessed[i][j]), (j + 1) * (40) + 13, i * 40 + 23);
                } else{ 
                    g.drawString(String.valueOf((int)labelsGuessed[i][j]), (j + 1) * (40) + 10, i * 40 + 23);
                }
            }
        }
        g.setColor(Color.BLACK);
        for (int i = 0; i < numClasses; i++) {
            g.drawLine(32, 20 + i * 40, 40, 20 + i * 40);
            if (i < 10) {
                g.drawString(String.valueOf(i), 15, 25 + i * 40);
            } else {
                g.drawString(String.valueOf(i), 9, 25 + i * 40);
            }
        }
        for (int i = 0; i < numClasses; i++) {
            g.drawLine(60 + i * 40, 40 * numClasses + 8, 60 + i * 40, 40 * numClasses);
            if (i < 10) {
                g.drawString(String.valueOf(i), 56 + i * 40, 40 * numClasses + 25);
            } else {
                g.drawString(String.valueOf(i), 52 + i * 40, 40 * numClasses + 25);
            }
        }
    }

    // Rotate an array clockwise by 90 degrees
    private static double[][] rotate90(double[][] arr) {
        int numRows = arr.length;
        int numCols = arr[0].length;

        int numThreads = Runtime.getRuntime().availableProcessors();
        int minSizeForParallel = 10000 * numThreads;
        double[][] result = new double[numCols][numRows];


        if (numRows * numCols >= minSizeForParallel) {
            ForkJoinPool pool = new ForkJoinPool(numThreads);
            List<Callable<Void>> tasks = new ArrayList<>();
            for (int i = 0; i < numRows; i++) {
                final int row = i;
                tasks.add(() -> {
                    for (int col = 0; col < numCols; col++) {
                        result[col][row] = arr[row][col];
                    }
                    return null;
                });
            }
            try {
                pool.invokeAll(tasks);
            } catch (Exception e) {
                Thread.currentThread().interrupt(); 
                e.printStackTrace();
            }
            pool.close();
        }  else {
            for (int row = 0; row < numRows; row++) {
                for (int col = 0; col < numCols; col++) {
                    result[col][row] = arr[row][col];
                }
            }
        }
        return result;
    }

    private int max(int[] arr) {
        int max = 0;
        for (int i : arr) {
            max = Math.max(max, i);
        }
        return max;
    }
}