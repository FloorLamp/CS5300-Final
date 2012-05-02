//Weighted Quick Union with path compression.

public class UnionFind {
	private int count; // number of connected components
	private int vertices; //number of total vertices
	
	ConnectedComponent[] components;
	ConnectedComponent[] stack;
	
	//initialize n components 0 through n-1
	public UnionFind(int n){
		count = n; //start with n separate components
		vertices = n;
		
		components = new ConnectedComponent[n];
		stack = new ConnectedComponent[n];

		
	}
	
	
	public int getComponentIDByLabel(int n){
		return find(n).getLabel();
	}
	
	
	public int edges(){
		return vertices - count;
	}
	
	public int vertices(){
		return vertices;
	}
	
	public int components(){
		return count;
	}
	

	public ConnectedComponent find(int a){
		ConnectedComponent cs = components[a];
		
		if (cs == null){
			ConnectedComponent c = new ConnectedComponent(a);
			c.child = new ConnectedComponent(a);
			c.child.parent = c;
			
			components[a] = c.child;
		}
		return find(components[a]);
	}
	
	public ConnectedComponent find(ConnectedComponent a){
		
		int top = 0;
		
		while(a.parent.child == null){
			stack[top++] = a;
			a = a.parent;
		}

		ConnectedComponent root = a;
		
		while (top > 0){
			a = stack[--top];
			a.parent = root;
		}
		
		return root.parent;
		
	}

	
	
	public boolean isConnected(int a, int b){
		return find(a) == find(b);
	}
	
	
	//merge two sets into one. smaller set gets merged under larger set.
	public void union(int a, int b){
		ConnectedComponent nA = find(a);
		ConnectedComponent nB = find(b);
		
		if (nA == nB)
			return;
		
		// Link the smaller set under the larger. 
		if (nA.numChildren < nB.numChildren){
			nA.child.parent = nB.child;
			nB.setID(nA.getID());
			nB.numChildren += nA.numChildren;

		} else {
			nB.child.parent = nA.child;
			nA.setID(nB.getID());
			nA.numChildren += nB.numChildren;
		}
		
		ConnectedComponent.compLabel(nA,nB);

		//number of components decreases by 1 after union
		count--;
		
		
	}
	
	   public static void main(String [] args)
	   {
		   int a = 10;
	      UnionFind x = new UnionFind(a);
	      x.union(1, 2);
	      x.union(3, 4);
	      x.union(1, 3);
	      x.union(6, 7);
	      x.union(7, 3);
	      x.union(6, 5);
	      
	      for(int i = 0; i < a; i++){
	    	  System.out.println("find "+ i + ":" + x.find(i));
	      }
	      
	      System.out.println("find " + ":" + x.find(2).numChildren);
	      
	   }
	   
	   
	
}
