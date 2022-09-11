package cz.cvut.fel.f_splay;

import java.io.*;
import java.util.StringTokenizer;

@SuppressWarnings("Duplicates")
public class Main {

    private int maxDepth = 0;
    private int modMaxDepth = 0;

    public static void main(String[] args) {
        new Main().run(null);
    }

    public void run(String file) {
        maxDepth = 0;
        modMaxDepth = 0;
        Node root = null;
        Node modRoot = null;

        //parsing input
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        if (file != null) {
            try {
                br = new BufferedReader(new FileReader(new File(file)));
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                System.exit(1);
            }
        }

        String line;
        StringTokenizer stringTokenizer;
        try {
            line = br.readLine();
            stringTokenizer = new StringTokenizer(line);
            int n = Integer.parseInt(stringTokenizer.nextToken());
            line = br.readLine();
            stringTokenizer = new StringTokenizer(line);
            for (int i = 0; i < n; i++) {
                int a = Integer.parseInt(stringTokenizer.nextToken());
                if (a > 0) {
                    root = insert(root, a);
                    modRoot = insertMod(modRoot, a);
                } else {
                    root = delete(root, -a);
                    modRoot = deleteMod(modRoot, -a);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        depth(root, 0);
        modDepth(modRoot, 0);
        System.out.println(maxDepth + " " + modMaxDepth);
    }

    // standard splay tree

    private Node insert(Node root, int k) {
        if (root == null) {
            return new Node(k);
        }
        Node node = insertt(root, k);
        // zig if needed
        if (node == null) {
            if (k < root.k) {
                if (root.l == null) {
                    root.l = new Node(k);
                } else {
                    root.l = insertt(root.l, k);
                }
                return rRot(root);
            } else {
                if (root.r == null) {
                    root.r = new Node(k);
                } else {
                    root.r = insertt(root.r, k);
                }
                return lRot(root);
            }
        }
        return node;
    }

    private Node insertt(Node node, int k) {
        if (k < node.k) {
            if (node.l == null) {
                return null;
            } else {
                Node n;
                // l-l
                if (k < node.l.k) {
                    if (node.l.l == null) {
                        n = new Node(k);
                    } else {
                        n = insertt(node.l.l, k);
                    }
                    if (n == null) {
                        return null;
                    } else {
                        node.l.l = n;
                        return rRot(rRot(node));
                    }
                } else { // l-r
                    if (node.l.r == null) {
                        n = new Node(k);
                    } else {
                        n = insertt(node.l.r, k);
                    }
                    if (n == null) {
                        return null;
                    } else {
                        node.l.r = n;
                        node.l = lRot(node.l);
                        return rRot(node);
                    }
                }
            }
        } else {
            if (node.r == null) {
                return null;
            } else {
                Node n;
                // r-r
                if (k > node.r.k) {
                    if (node.r.r == null) {
                        n = new Node(k);
                    } else {
                        n = insertt(node.r.r, k);
                    }
                    if (n == null) {
                        return null;
                    } else {
                        node.r.r = n;
                        return lRot(lRot(node));
                    }
                } else { // r-l
                    if (node.r.l == null) {
                        n = new Node(k);
                    } else {
                        n = insertt(node.r.l, k);
                    }
                    if (n == null) {
                        return null;
                    } else {
                        node.r.l = n;
                        node.r = rRot(node.r);
                        return lRot(node);
                    }
                }
            }
        }
    }

    private Node splay(Node node, int k) {
        if (k < node.k) {
            if (node.l == null || node.l.k == k) {
                return null;
            } else {
                Node n;
                // l-l
                if (k < node.l.k) {
                    if (node.l.l.k != k) {
                        n = splay(node.l.l, k);
                        if (n == null) {
                            return null;
                        }
                        node.l.l = n;
                    }
                    return rRot(rRot(node));
                } else { // l-r
                    if (node.l.r.k != k) {
                        n = splay(node.l.r, k);
                        if (n == null) {
                            return null;
                        }
                        node.l.r = n;
                    }
                    node.l = lRot(node.l);
                    return rRot(node);
                }
            }
        } else {
            if (node.r == null || node.r.k == k) {
                return null;
            } else {
                Node n;
                // r-r
                if (k > node.r.k) {
                    if (node.r.r.k != k) {
                        n = splay(node.r.r, k);
                        if (n == null) {
                            return null;
                        }
                        node.r.r = n;
                    }
                    return lRot(lRot(node));
                } else { // r-l
                    if (node.r.l.k != k) {
                        n = splay(node.r.l, k);
                        if (n == null) {
                            return null;
                        }
                        node.r.l = n;
                    }
                    node.r = rRot(node.r);
                    return lRot(node);
                }
            }
        }
    }

    private Node splayMax(Node node) {
        if (node.r == null) {
            return node;
        }
        if (node.r.r == null) {
            return null;
        } else {
            Node n = splayMax(node.r.r);
            if (n == null) {
                return null;
            }
            node.r.r = n;
        }
        return lRot(lRot(node));
    }

    private Node delete(Node root, int k) {
        if (root.k != k) {
            Node n = splay(root, k);
            if (n == null) {
                if (k < root.k) {
                    if (root.l.k != k) {
                        root.l = splay(root.l, k);
                    }
                    root = rRot(root);
                } else {
                    if (root.r.k != k) {
                        root.r = splay(root.r, k);
                    }
                    root = lRot(root);
                }
            } else {
                root = n;
            }
        }
        if (root.l == null) {
            return root.r;
        }
        Node n = splayMax(root.l);
        if (n == null) {
            root.l.r = splayMax(root.l.r);
            root.l = lRot(root.l);
        } else {
            root.l = n;
        }
        root.l.r = root.r;
        return root.l;
    }

    private void depth(Node node, int depth) {
        if (node.l == null && node.r == null) {
            maxDepth = depth > maxDepth ? depth : maxDepth;
            return;
        }
        if (node.l != null) {
            depth(node.l, depth + 1);
        }
        if (node.r != null) {
            depth(node.r, depth + 1);
        }
    }


    // zig only tree

    private Node insertMod(Node root, int k) {
        if (root == null) {
            return new Node(k);
        } else {
            return insertModd(root, k);
        }
    }

    private Node insertModd(Node node, int k) {
        if (k < node.k) {
            if (node.l == null) {
                node.l = new Node(k);
            } else {
                node.l = insertModd(node.l, k);
            }
            return rRot(node);
        } else {
            if (node.r == null) {
                node.r = new Node(k);
            } else {
                node.r = insertModd(node.r, k);
            }
            return lRot(node);
        }
    }

    private Node splayMod(Node node, int k) {
        if (k < node.k) {
            if (node.l == null) {
                return node;
            }
            node.l = splayMod(node.l, k);
            return rRot(node);
        } else if (k == node.k) {
            return node;
        } else {
            if (node.r == null) {
                return node;
            }
            node.r = splayMod(node.r, k);
            return lRot(node);
        }
    }

    private Node splayMaxMod(Node node) {
        if (node.r == null) {
            return node;
        }
        node.r = splayMaxMod(node.r);
        return lRot(node);
    }

    private Node deleteMod(Node root, int k) {
        Node n = splayMod(root, k);
        if (n.l == null) {
            return n.r;
        }
        n.l = splayMaxMod(n.l);
        n.l.r = n.r;
        return n.l;
    }

    private void modDepth(Node node, int depth) {
        if (node.l == null && node.r == null) {
            modMaxDepth = depth > modMaxDepth ? depth : modMaxDepth;
            return;
        }
        if (node.l != null) {
            modDepth(node.l, depth + 1);
        }
        if (node.r != null) {
            modDepth(node.r, depth + 1);
        }
    }


    // rotations, node def

    private Node lRot(Node node) {
        Node r = node.r;
        node.r = r.l;
        r.l = node;
        return r;
    }

    private Node rRot(Node node) {
        Node l = node.l;
        node.l = l.r;
        l.r = node;
        return l;
    }

    class Node {
        int k;
        Node l;
        Node r;

        Node(int k) {
            this.k = k;
        }
    }
}
