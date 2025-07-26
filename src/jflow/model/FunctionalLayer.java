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
    
    protected abstract Layer[] defineLayers();

    public abstract JMatrix forward(JMatrix input, boolean training);
    public abstract JMatrix backward(JMatrix input);

    public abstract int[] outputShape();

    public Layer[] getLayers() {
        return components;
    }

    @Override
    public JMatrix[] forwardDebugData() {
        int length = 0;
        for (Layer l : components) {
            if (!(l instanceof FunctionalLayer)) {
                length += l.forwardDebugData().length;
            }
        }
        JMatrix[] debugData = new JMatrix[length];
        int index = 0;
        for (Layer l : components) {
            if (!(l instanceof FunctionalLayer)) {
                for (JMatrix matrix : l.forwardDebugData()) {
                    debugData[index++] = matrix;
                }
            }
        }
        return debugData;
    }
    
    @Override
    public JMatrix[] backwardDebugData() {
        int length = 0;
        for (Layer l : components) {
            if (!(l instanceof FunctionalLayer)) {
                length += l.backwardDebugData().length;
            }
        }
        JMatrix[] debugData = new JMatrix[length];
        int index = 0;
        for (Layer l : components) {
            if (!(l instanceof FunctionalLayer)) {
                for (JMatrix matrix : l.backwardDebugData()) {
                    debugData[index++] = matrix;
                }
            }
        }
        return debugData;
    }

    @Override
    public void printForwardDebug() {
        for (Layer l : components) {
            if (l instanceof FunctionalLayer) {
                l.printForwardDebug();
            }
        }
        if (forwardDebugData() != null) {
            Callbacks.printStats(
            getName() + " output",
            forwardDebugData()
        );
        }

        
    }

    @Override
    public void printBackwardDebug() {
        for (Layer l : components) {
            if (l instanceof FunctionalLayer) {
                l.printBackwardDebug();
            }
        }
        if (backwardDebugData() != null) {
            Callbacks.printStats(
            getName() + " gradients",
            backwardDebugData()
        );
        }
    }
}
