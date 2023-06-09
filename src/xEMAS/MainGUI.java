

import javax.swing.JFrame;

/**
 * Main class
 */
public class MainGUI extends JFrame {

    //protected BuilderJInternalFrame m_BuilderView = null;
    protected PredictorJInternalFrame m_Predictor = null;
    protected AboutJInternalFrame m_About = null;
   
    /** Creates new form Main */
    public MainGUI() {
        initComponents();
        setSize(900, 700);
        setTitle("Epitope Prediction Software");
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        theDesktop = new javax.swing.JDesktopPane();
        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jLabel1 = new javax.swing.JLabel();
        jLabel4 = new javax.swing.JLabel();
        jMenuBar1 = new javax.swing.JMenuBar();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        getContentPane().add(theDesktop, java.awt.BorderLayout.CENTER);
        theDesktop.setBackground(new java.awt.Color(0, 0, 0));
        theDesktop.setBorder(javax.swing.BorderFactory.createCompoundBorder());
        theDesktop.setAutoscrolls(true);
        theDesktop.setCursor(new java.awt.Cursor(java.awt.Cursor.DEFAULT_CURSOR));
        theDesktop.setDebugGraphicsOptions(javax.swing.DebugGraphics.BUFFERED_OPTION);
        theDesktop.setDoubleBuffered(true);
        theDesktop.setFocusTraversalPolicyProvider(true);
        theDesktop.setInheritsPopupMenu(true);
        theDesktop.setName("Epitope Prediction"); // NOI18N

        jButton1.setBackground(new java.awt.Color(0, 51, 102));
        jButton1.setFont(new java.awt.Font("Tahoma", 0, 18)); // NOI18N
        jButton1.setForeground(new java.awt.Color(204, 0, 0));
        jButton1.setText("Start Epitopes Prediction");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });
        jButton1.setBounds(320, 420, 250, 60);
        theDesktop.add(jButton1, javax.swing.JLayeredPane.DEFAULT_LAYER);

        jButton2.setBackground(new java.awt.Color(0, 51, 102));
        jButton2.setForeground(new java.awt.Color(204, 0, 0));
        jButton2.setText("About");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });
        jButton2.setBounds(390, 490, 100, 30);
        theDesktop.add(jButton2, javax.swing.JLayeredPane.DEFAULT_LAYER);

        jLabel1.setIcon(new javax.swing.ImageIcon(getClass().getResource("/YoussefBakr_1.jpg"))); // NOI18N
        jLabel1.setFocusTraversalPolicyProvider(true);
        jLabel1.setBounds(200, 50, 480, 360);
        theDesktop.add(jLabel1, javax.swing.JLayeredPane.DEFAULT_LAYER);

        jLabel4.setFont(new java.awt.Font("Tahoma", 0, 12)); // NOI18N
        jLabel4.setForeground(new java.awt.Color(204, 0, 0));
        jLabel4.setText("Epitopes Prediction Software ");
        jLabel4.setBounds(360, 20, 170, 15);
        theDesktop.add(jLabel4, javax.swing.JLayeredPane.DEFAULT_LAYER);

        getContentPane().add(theDesktop, java.awt.BorderLayout.CENTER);
        setJMenuBar(jMenuBar1);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        // TODO add your handling code here:
                    m_Predictor = new PredictorJInternalFrame();
            theDesktop.add(m_Predictor);
            m_Predictor.setVisible(true);
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        // TODO add your handling code here:
        m_About = new AboutJInternalFrame();
            theDesktop.add(m_About);
            m_About.setVisible(true);
    }//GEN-LAST:event_jButton2ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        java.awt.EventQueue.invokeLater(new Runnable() {

            public void run() {
                new MainGUI().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel4;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JDesktopPane theDesktop;
    // End of variables declaration//GEN-END:variables
}
