public class practice2 {
	public static void main(String[] args) {
		//System.err.close();
		
		//DetectOutliers dol = new DetectOutliers("C:\\Users\\ZK0GJXO\\Documents\\Files.txt");
		//dol.printAnomalies();
		
		/*try {
			PrintWriter writer = new PrintWriter("Training.txt");
			writer.println("ID\tREF\tinsID\tBook\tCash1\tCash2\tanomalous");
			for(int i = 1; i <= 50; i++){
				double random = Math.random();
				String REF = "CSHP";
				String insID = "OTHR";
				String book = "CNY";
				int anomalous = 0;
				
				int id = (int) ((random*9 + 1) * 10000);
				int cash1 = (int) ((Math.random()*7 + 2)* 100000);
				int cash2 = (int) ((Math.random()*7 + 3)* 100000);
				
				if(cash1 > cash2)
					anomalous = 1;
				
				writer.println(id + "\t" + REF + "\t" + insID + "\t" + book + "\t" + cash1 + "\t" + cash2 + "\t" + anomalous);
			
			}
			
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
		

		/*try {
			PrintWriter writer = new PrintWriter("Test2.txt");
			writer.println("ID\tREF\tinsID\tBook\tCash1\tExpDate");
			for(int i = 1; i <= 50; i++){
				double random = Math.random();
				String REF = "CSHP";
				String insID = "OTHR";
				String book = "CNY";
				int id = (int) ((random*9 + 1) * 10000);
				int cash1 = (int) ((Math.random()*7 + 2)* 100000);

				writer.println(id + "\t" + REF + "\t" + insID + "\t" + book + "\t" + cash1 + "\t" + "2017-09-20 00:00:00");
			
			}
			
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
		
		try {
			PrintWriter writer = new PrintWriter("Files.txt");
			writer.println("ID\tSize\tVolume\tNetCash");
			for(int i = 1; i <= 200; i++){
				double random = Math.random();
				int id = (int) ((random*9 + 1) * 10000);
				int size = (int)((random*9 + 1)*10);
				int volume = (int)((random+1)*1000*1.01);
				double cash = Math.round((random + 1)*10000*1.04*100)/100.0;

				writer.println(id + "\t" + size + "\t" + volume + "\t" + cash);
			
			}
			
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
