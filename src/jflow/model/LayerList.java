package jflow.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class LayerList {
    // Node data structure
    private static class LayerNode {
        final Layer layer;
        final List<LayerNode> children = new ArrayList<>();
        LayerNode parent;
        int level;
        
        LayerNode(Layer layer) {
            this.layer = layer;
        }
        
        void addChild(LayerNode child) {
            children.add(child);
            child.parent = this;
            child.level = this.level + 1;
        }
    }
    
    private final Map<Layer, LayerNode> nodeMap = new LinkedHashMap<>(); // Insertion order must be preserved
    private final List<LayerNode> roots = new ArrayList<>(); // Top-level layers
    private final Map<String, Integer> typeCounts = new HashMap<>();
    
    protected LayerList() {}
    
    protected void add(Layer layer) {
        // Externally added layers are roots
        LayerNode node = processLayer(layer, null);
        roots.add(node);
        
        if (layer instanceof FunctionalLayer functional) {
            // Handle internal layers as children
            processFunctionalLayer(functional, node);
        }
        layer.link(this);
    }
    
    protected List<Layer> getLevel(int level) {
        return nodeMap.values().stream()
            .filter(node -> node.level == level)
            .map(node -> node.layer)
            .collect(Collectors.toList());
    }
    
    protected List<Layer> getFlat() {
        // Return layers in insertion order
        return nodeMap.keySet().stream().collect(Collectors.toList());
    }
    
    public Layer getFirst() {
        return nodeMap.isEmpty() ? null : nodeMap.keySet().iterator().next();
    }
    
    public Layer getLast() {
        if (nodeMap.isEmpty()) return null;
        
        Layer last = null;
        for (Layer layer : nodeMap.keySet()) {
            last = layer;
        }
        return last;
    }
    
    @SuppressWarnings("unchecked")
    public <T extends Layer> List<T> getLayersOfType(Class<T> type) {
        return nodeMap.keySet().stream()
            .filter(type::isInstance)
            .map(layer -> (T) layer)
            .collect(Collectors.toList());
    }
    
    
    public List<Layer> getChildren(Layer layer) {
        LayerNode node = nodeMap.get(layer);
        if (node == null) return new ArrayList<>();
        
        return node.children.stream()
            .map(child -> child.layer)
            .collect(Collectors.toList());
    }
    
    public Layer getParent(Layer layer) {
        LayerNode node = nodeMap.get(layer);
        return (node != null && node.parent != null) ? node.parent.layer : null;
    }
    
    public int getDepth(Layer layer) {
        LayerNode node = nodeMap.get(layer);
        return node != null ? node.level : -1;
    }
    
    public int getMaxDepth() {
        return nodeMap.values().stream()
            .mapToInt(node -> node.level)
            .max()
            .orElse(-1);
    }
    
    public boolean contains(Layer layer) {
        return nodeMap.containsKey(layer);
    }
    
    public List<Layer> getPath(Layer layer) {
        LayerNode node = nodeMap.get(layer);
        if (node == null) return new ArrayList<>();
        
        List<Layer> path = new ArrayList<>();
        LayerNode current = node;
        while (current != null) {
            path.add(0, current.layer); // Add to front
            current = current.parent;
        }
        return path;
    }
    
    
    private void nameLayer(Layer layer) {
        String type = layer.getType();
        int count = typeCounts.getOrDefault(type, 0);
        typeCounts.put(type, count + 1);
        layer.setName(type + "_" + (count + 1));
    }
    
    private LayerNode processLayer(Layer layer, LayerNode parent) {
        nameLayer(layer);
        
        LayerNode node = new LayerNode(layer);
        node.level = parent != null ? parent.level + 1 : 0;
        
        nodeMap.put(layer, node);
        
        if (parent != null) {
            parent.addChild(node);
        }
        
        return node;
    }
    
    private void processFunctionalLayer(FunctionalLayer functional, LayerNode parentNode) {
        for (Layer layer : functional.getLayers()) {
            LayerNode childNode = processLayer(layer, parentNode);
            layer.link(this);
            if (layer instanceof FunctionalLayer internalFunctional) {
                processFunctionalLayer(internalFunctional, childNode);
            }
        }
    }
}