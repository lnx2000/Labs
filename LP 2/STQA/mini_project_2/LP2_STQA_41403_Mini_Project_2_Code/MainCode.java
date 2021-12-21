import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.interactions.Actions;


public class MainCode {
	
	static String fname = "Tony";
	static String lname = "Stark";
	static String email = "ironman@starkinductries.com";
	static char gender = 'M';
	static int mobileNo = 1234567890;
	static String dob = "24 Jan 1998";
	static List<String> hobbies = Arrays.asList("Sports","Reading");
	static String currentAddress = "India";
	public static void main(String[] args) throws InterruptedException {
		System.setProperty("webdriver.chrome.driver","D:\\chromedriver.exe");
		
		// Initiate Chrome Driver Class
		WebDriver driver = (WebDriver) new ChromeDriver();
		
		// Maximize the browser
		driver.manage().window().maximize();
		
		// Launch Website
		driver.navigate().to("https://demoqa.com/automation-practice-form");
		Thread.sleep(2000);
		
		Actions action = new Actions(driver);
		
		driver.findElement(By.xpath("//*[@id=\"firstName\"]")).sendKeys(fname);
		driver.findElement(By.xpath("//*[@id=\"lastName\"]")).sendKeys(lname);
		driver.findElement(By.xpath("//*[@id=\"userEmail\"]")).sendKeys(email);
		
		// Gender
		if(gender == 'M')
			action.moveToElement(driver.findElement(By.xpath("//*[@id=\"gender-radio-1\"]"))).click().perform();
		else if(gender == 'F')
			action.moveToElement(driver.findElement(By.xpath("//*[@id=\"gender-radio-2\"]"))).click().perform();
		else
			action.moveToElement(driver.findElement(By.xpath("//*[@id=\"gender-radio-3\"]"))).click().perform();
		
		// Mobile No
		driver.findElement(By.xpath("//*[@id=\"userNumber\"]")).sendKeys(Integer.toString(mobileNo));
	
		// Hobbies
		WebElement element = driver.findElement(By.xpath("//*[@id=\"hobbiesWrapper\"]/div[2]"));
		List<WebElement> elementList = element.findElements(By.tagName("div"));
		for(int i=0;i<hobbies.size();i++) {
			for(int j=0;j<elementList.size();j++) {
				WebElement label = elementList.get(j).findElement(By.tagName("label"));
				if(label.getText().equalsIgnoreCase(hobbies.get(i))) {
					action.moveToElement(elementList.get(j).findElement(By.tagName("input"))).click().perform();
				}
			}
		}
		
		
		// Current address
		driver.findElement(By.xpath("//*[@id=\"currentAddress\"]")).sendKeys(currentAddress);
		
		// Submit button
		driver.findElement(By.xpath("//*[@id=\"submit\"]")).submit();
		System.out.println("Done");
		
	}
}
