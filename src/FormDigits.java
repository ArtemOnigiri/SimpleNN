import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

public class FormDigits extends JFrame implements Runnable, MouseListener, MouseMotionListener, KeyListener {

    private final int w = 28;
    private final int h = 28;
    private final int scale = 32;

    private int mousePressed = 0;
    private int mx = 0;
    private int my = 0;
    private double[][] colors = new double[w][h];

    private BufferedImage img = new BufferedImage(w * scale + 200, h * scale, BufferedImage.TYPE_INT_RGB);
    private BufferedImage pimg = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    private int frame = 0;

    private NeuralNetwork nn;

    public FormDigits(NeuralNetwork nn) {
        this.nn = nn;
        this.setSize(w * scale + 200 + 16, h * scale + 38);
        this.setVisible(true);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setLocation(50, 50);
        this.add(new JLabel(new ImageIcon(img)));
        addMouseListener(this);
        addMouseMotionListener(this);
        addKeyListener(this);
    }

    @Override
    public void run() {
        while (true) {
            this.repaint();
//            try { Thread.sleep(17); } catch (InterruptedException e) {}
        }
    }

    @Override
    public void paint(Graphics g) {
        double[] inputs = new double[784];
        for (int i = 0; i < w; i++) {
            for (int j = 0; j < h; j++) {
                if(mousePressed != 0) {
                    double dist = (i - mx) * (i - mx) + (j - my) * (j - my);
                    if(dist < 1) dist = 1;
                    dist *= dist;
                    if(mousePressed == 1) colors[i][j] += 0.1 / dist;
                    else colors[i][j] -= 0.1 / dist;
                    if (colors[i][j] > 1) colors[i][j] = 1;
                    if (colors[i][j] < 0) colors[i][j] = 0;
                }
                int color = (int)(colors[i][j] * 255);
                color = (color << 16) | (color << 8) | color;
                pimg.setRGB(i, j, color);
                inputs[i + j * w] = colors[i][j];
            }
        }
        double[] outputs = nn.feedForward(inputs);
        int maxDigit = 0;
        double maxDigitWeight = -1;
        for (int i = 0; i < 10; i++) {
            if(outputs[i] > maxDigitWeight) {
                maxDigitWeight = outputs[i];
                maxDigit = i;
            }
        }
        Graphics2D ig = (Graphics2D) img.getGraphics();
        ig.drawImage(pimg, 0, 0, w * scale, h * scale, this);
        ig.setColor(Color.lightGray);
        ig.fillRect(w * scale + 1, 0, 200, h * scale);
        ig.setFont(new Font("TimesRoman", Font.BOLD, 48));
        for (int i = 0; i < 10; i++) {
            if(maxDigit == i) ig.setColor(Color.RED);
            else ig.setColor(Color.GRAY);
            ig.drawString(i + ":", w * scale + 20, i * w * scale / 15 + 150);
            Color rectColor = new Color(0, (float)outputs[i], 0);
            int rectWidth = (int)(outputs[i] * 100);
            ig.setColor(rectColor);
            ig.fillRect(w * scale + 70, i * w * scale / 15 + 122, rectWidth, 30);
        }
        g.drawImage(img, 8, 30, w * scale + 200, h * scale, this);
        frame++;
    }

    @Override
    public void mouseClicked(MouseEvent e) {

    }

    @Override
    public void mousePressed(MouseEvent e) {
        mousePressed = 1;
        if(e.getButton() == 3) mousePressed = 2;
    }

    @Override
    public void mouseReleased(MouseEvent e) {
        mousePressed = 0;
    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {

    }

    @Override
    public void keyTyped(KeyEvent e) {

    }

    @Override
    public void keyPressed(KeyEvent e) {
        if(e.getKeyCode() == KeyEvent.VK_SPACE) {
            colors = new double[w][h];
        }
    }

    @Override
    public void keyReleased(KeyEvent e) {

    }

    @Override
    public void mouseDragged(MouseEvent e) {
        mx = e.getX() / scale;
        my = e.getY() / scale;
    }

    @Override
    public void mouseMoved(MouseEvent e) {
        mx = e.getX() / scale;
        my = e.getY() / scale;
    }
}