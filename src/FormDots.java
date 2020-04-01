import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.ArrayList;
import java.util.function.UnaryOperator;

public class FormDots extends JFrame implements Runnable, MouseListener {

    private final int w = 1280;
    private final int h = 720;

    private BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    private BufferedImage pimg = new BufferedImage(w / 8, h / 8, BufferedImage.TYPE_INT_RGB);
    private int frame = 0;

    private NeuralNetwork nn;

    public List<Point> points = new ArrayList<>();

    public FormDots() {
        UnaryOperator<Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
        UnaryOperator<Double> dsigmoid = y -> y * (1 - y);
        nn = new NeuralNetwork(0.01, sigmoid, dsigmoid, 2, 5, 5, 2);

        this.setSize(w + 16, h + 38);
        this.setVisible(true);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setLocation(50, 50);
        this.add(new JLabel(new ImageIcon(img)));
        addMouseListener(this);
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
        if(points.size() > 0) {
            for (int k = 0; k < 10000; k++) {
                Point p = points.get((int) (Math.random() * points.size()));
                double nx = (double) p.x / w - 0.5;
                double ny = (double) p.y / h - 0.5;
                nn.feedForward(new double[]{nx, ny});
                double[] targets = new double[2];
                if (p.type == 0) targets[0] = 1;
                else targets[1] = 1;
                nn.backpropagation(targets);
            }
        }
        for (int i = 0; i < w / 8; i++) {
            for (int j = 0; j < h / 8; j++) {
                double nx = (double) i / w * 8 - 0.5;
                double ny = (double) j / h * 8 - 0.5;
                double[] outputs = nn.feedForward(new double[]{nx, ny});
                double green = Math.max(0, Math.min(1, outputs[0] - outputs[1] + 0.5));
                double blue = 1 - green;
                green = 0.3 + green * 0.5;
                blue = 0.5 + blue * 0.5;
                int color = (100 << 16) | ((int)(green * 255) << 8) | (int)(blue * 255);
                pimg.setRGB(i, j, color);
            }
        }
        Graphics ig = img.getGraphics();
        ig.drawImage(pimg, 0, 0, w, h, this);
        for (Point p : points) {
            ig.setColor(Color.WHITE);
            ig.fillOval(p.x - 3, p.y - 3, 26, 26);
            if (p.type == 0) ig.setColor(Color.GREEN);
            else ig.setColor(Color.BLUE);
            ig.fillOval(p.x, p.y, 20, 20);
        }
        g.drawImage(img, 8, 30, w, h, this);
        frame++;
    }

    @Override
    public void mouseClicked(MouseEvent e) {

    }

    @Override
    public void mousePressed(MouseEvent e) {
        int type = 0;
        if(e.getButton() == 3) type = 1;
        points.add(new Point(e.getX() - 16, e.getY() - 38, type));
    }

    @Override
    public void mouseReleased(MouseEvent e) {

    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {

    }
}