package jflow.model;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;
import jflow.utils.AnsiCodes;
import jflow.utils.Callbacks;

public abstract class FunctionalLayer extends ShapeAlteringLayer {
    private Layer[] components;
    private static final int DEBUG_SHELL_WIDTH = 54;

    public FunctionalLayer(String type) {
        super(type);
    }
    
    protected abstract Layer[] defineLayers();

    public Layer[] getLayers() {
        components = defineLayers();
        return components;
    }

    @Override
    protected int numTrainableParameters() {
        int paramCount = 0;
        for (Layer l : components) {
            paramCount += l.numTrainableParameters();
        }
        return paramCount;
    }

    @Override
    public void printForwardDebug() {
        printLayeredDebug(getName(), true);
    }

    @Override
    public void printBackwardDebug() {
        printLayeredDebug(getName(), false);
    }

    // Recursively nest FunctionalLayer debug shells
    private void printLayeredDebug(String debugTitle, boolean forward) {
        PrintStream originalOut = System.out;
        boolean createdNewStream = false;
        
        if (!(System.out instanceof IndentingPrintStream)) {
            IndentingPrintStream stream = new IndentingPrintStream(originalOut);
            System.setOut(stream);
            createdNewStream = true;
        }
        
        printDebugShellTop(debugTitle);
        IndentingPrintStream.addIndent(AnsiCodes.YELLOW + "│ ");
        
        // Printout order is reversed between the forward pass and backward pass
        for (
            int index = forward ? 0 : components.length - 1;
            forward ? index <= components.length - 1 : index >= 0;  
            index += forward ? 1 : -1
        ) {
            if (forward) {
                components[index].printForwardDebug();
            } else {
                components[index].printBackwardDebug();
            }
        }
        
        JMatrix[] debugData = (forward) ? forwardDebugData() : backwardDebugData();
        if (debugData != null) {
            Callbacks.printStats("", debugData);
        }
        
        IndentingPrintStream.removeIndent(AnsiCodes.YELLOW + "│ ");
        printDebugShellBottom();

        
        if (createdNewStream) {
            System.setOut(originalOut);
        }
    }

    private void printDebugShellTop(String debugTitle) {
        int topBracketWidth = DEBUG_SHELL_WIDTH;

        if (!debugTitle.equals("")) {
            debugTitle = " " + debugTitle + " ";
        }
        topBracketWidth -= debugTitle.length();
        boolean oddWidth = topBracketWidth % 2 != 0;
        topBracketWidth /= 2;
        System.out.print(AnsiCodes.YELLOW + "╭" + "─".repeat(topBracketWidth));
        if (oddWidth) {
            System.out.print("─");
        }
        System.out.print(
            AnsiCodes.BOLD + debugTitle + 
            AnsiCodes.RESET + AnsiCodes.YELLOW
            + "─".repeat(topBracketWidth)
        );
        
        System.out.println("╮");
    }

    private void printDebugShellBottom() {
        System.out.print(
            AnsiCodes.YELLOW + "╰"
            + "─".repeat(DEBUG_SHELL_WIDTH)
        );
        System.out.println("╯" + AnsiCodes.RESET);
    }

    // Custom output stream to print lines with a specified indent
    private static class IndentingPrintStream extends PrintStream {
        private static final ThreadLocal<String> indent = ThreadLocal.withInitial(() -> "");
        private boolean atLineStart = true;

        public IndentingPrintStream(OutputStream out) {
            super(out, true);
        }

        public static void addIndent(String extra) {
            indent.set(indent.get() + extra);
        }

        public static void removeIndent(String extra) {
            String current = indent.get();
            if (current.endsWith(extra)) {
                indent.set(current.substring(0, current.length() - extra.length()));
            }
        }

        @Override
        public void write(byte[] buf, int off, int len) {
            String str = new String(buf, off, len);
            String[] lines = str.split("\n", -1); // Keep empty lines
            
            for (int i = 0; i < lines.length; i++) {
                try {
                    // Only add indent at the start of a line
                    if (atLineStart && !lines[i].isEmpty()) {
                        out.write(indent.get().getBytes());
                    }
                    out.write(lines[i].getBytes());
                    
                    if (i < lines.length - 1) {
                        out.write('\n');
                        atLineStart = true; 
                    } else {
                        atLineStart = false;
                    }
                } catch (IOException e) {
                    setError();
                }
            }
            
            if (str.endsWith("\n")) {
                atLineStart = true;
            }
        }
    }

}
