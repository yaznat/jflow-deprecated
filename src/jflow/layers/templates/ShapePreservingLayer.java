package jflow.layers.templates;

import jflow.model.Layer;

public abstract class ShapePreservingLayer extends Layer{
    public ShapePreservingLayer(String type) {
        super(type, false);
    }

    @Override
    public int[] outputShape() {
        return getPreviousLayer().outputShape();
    }    
}
