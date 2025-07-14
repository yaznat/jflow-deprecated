package jflow.model;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;
import jflow.utils.Callbacks;

public abstract class FunctionalLayer extends ShapeAlteringLayer {
    private Layer[] components;
    public FunctionalLayer(String name) {
        super(name);
    }

    @Override
    public void build(int IDnum) {
        super.build(IDnum);
        this.components = defineLayers();
        for (Layer l : components) {
            l.setEnclosingLayer(this);
        }
    }
    
    public abstract Layer[] defineLayers();

    public abstract JMatrix forward(JMatrix input, boolean training);
    public abstract JMatrix backward(JMatrix input);

    public abstract int[] outputShape();

    public Layer[] getLayers() {
        return components;
    }

    public void printForwardDebug() {
        int length = 0;
        for (Layer l : components) {
            for (JMatrix matrix : l.forwardDebugData()) {
                length++;
            }
        }
        JMatrix[] debugData = new JMatrix[length];
        int index = 0;
        for (Layer l : components) {
            for (JMatrix matrix : l.forwardDebugData()) {
                debugData[index++] = matrix;
            }
        }

        Callbacks.printStats(
            getName() + " output",
            debugData
        );
    }

    public void printBackwardDebug() {
        int length = 0;
        for (Layer l : components) {
            for (JMatrix matrix : l.backwardDebugData()) {
                length++;
            }
        }
        JMatrix[] debugData = new JMatrix[length];
        int index = 0;
        for (Layer l : components) {
            for (JMatrix matrix : l.backwardDebugData()) {
                debugData[index++] = matrix;
            }
        }

        Callbacks.printStats(
            getName() + " gradients",
            debugData
        );
    }
}
