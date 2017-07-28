/*try {
			PrintWriter writer = new PrintWriter("C:\\Users\\ZK0GJXO\\Documents\\Training.txt");
			writer.println("ID\tREF\tinsID\tBook\tCash\tAnomalous");
			for(int i = 1; i <= 50; i++){
				double random = Math.random();
				String REF = "CSHP";
				String insID = "OTHR";
				String book = "CNY";
				int id = (int) ((random*9 + 1) * 10000);
				int cashflow = (int) ((random*7 + 3)* 1000000);
				int anomalous = 0;
				
				if(i%11 == 0){
					 REF = "CSHE";
					 book = "SNY";
					 anomalous = 1;
				}
				
				writer.println(id + "\t" + REF + "\t" + insID + "\t" + book + "\t" + cashflow + "\t" + anomalous);
			
			}
			
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
		

		try {
			PrintWriter writer = new PrintWriter("C:\\Users\\ZK0GJXO\\Documents\\Training.txt");
			writer.println("ID\tREF\tinsID\tBook\tCash");
			for(int i = 1; i <= 50; i++){
				double random = Math.random();
				String REF = "CSHP";
				String insID = "OTHR";
				String book = "CNY";
				int id = (int) ((random*9 + 1) * 10000);
				int cashflow = (int) ((random*7 + 3)* 1000000);
				
				if(i%11 == 0){
					 REF = "CSHE";
					 book = "SNY";
				}
				
				writer.println(id + "\t" + REF + "\t" + insID + "\t" + book + "\t" + cashflow);
			
			}
			
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
