package jflow.utils;

import java.awt.Color;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JPanel;

import jflow.data.JMatrix;

class ImageDisplay extends JPanel{

    private JFrame frame;
    private JMatrix image;
    private int scaleFactor;

    protected ImageDisplay(JMatrix image, int scaleFactor, String label) {
        this.image = image;
        int height = image.height();
        int width = image.width();


        // Since JFrame has a minimum visual width > 0
        if (height < 100 && scaleFactor < 2) {
            scaleFactor = 2;
        }

        this.scaleFactor = scaleFactor;

        frame = new JFrame();
        frame.setBounds(0, 0, width * scaleFactor, height * scaleFactor + 25);
        frame.setResizable(false);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setTitle(label);
        frame.add(this);
        repaint();
        frame.setVisible(true);

    }

    public void paintComponent(Graphics g) {
        int channel1; int channel2; int channel3;
        // RGB
        if (image.channels() == 3) {
            channel1 = 0;
            channel2 = 1;
            channel3 = 2;
        } 
        // grayscale
        else {
            channel1 = channel2 = channel3 = 0;
        }
        for (int i = 0; i < image.height(); i++) {
            for (int j = 0; j < image.width(); j++) {
                g.setColor(new Color(
                    (int)image.get(0, channel1, i, j), 
                    (int)image.get(0, channel2, i, j), 
                    (int)image.get(0, channel3, i, j)
                    )
                );
                g.fillRect(i * scaleFactor, j * scaleFactor, scaleFactor, scaleFactor);
            }
        }
    }
}
