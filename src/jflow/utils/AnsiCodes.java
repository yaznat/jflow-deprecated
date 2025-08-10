package jflow.utils;

public final class AnsiCodes {
    public final static String TEAL = "\033[38;2;0;153;153m";
    public final static String YELLOW = "\033[38;2;222;197;15m";
    public final static String ORANGE = "\033[38;2;255;165;0m";
    public final static String DARK_ORANGE = "\033[38;2;255;150;0m";
    public final static String BLUE = "\033[94m";
    public final static String WHITE = "\033[37m";
    public final static String GREEN = "\033[38;2;0;204;0m";
    public final static String RED = "\033[38;2;255;0;0m"; 
    public final static String PURPLE = "\033[35m";
    public final static String GRAY = "\033[38;2;60;60;60m";
    public final static String BOLD = "\033[1m";
    public final static String ITALIC = "\033[3m";
    public final static String RESET = "\033[0m";

    private AnsiCodes(){} // Prevent instantiation
}
