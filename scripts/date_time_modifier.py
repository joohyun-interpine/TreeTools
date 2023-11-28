import laspy
from datetime import datetime, timedelta

class DateTimeAdjuster:
    """
    The given date can be converted into New Zealand Date Daylight Time ftom Greenwich Mean Time <GMT to NZDT>,
    New Zealand is a country that uses a daylight system which is saving an hour within a specific season,
    so using GMT requires opt out the offset between two different time system depending on when.    
    """
    def __init__(self) -> None:
        pass        
    
    def get_daylight_saving_starts_date(self, year):
        """
        Daylight saving starts each year at 2am on the last Sunday in September

        Args:
            year (int): year 'yyyy' that you want to find a NZDT date

        Returns:
            a class 'datetime.datetime'
        """
        # Find the last day of September for the given year
        last_day_of_september = datetime(year, 10, 1) - timedelta(days=1)
        # Calculate the difference in days to find the last Sunday
        days_until_last_sunday = (last_day_of_september.weekday() - 6) % 7
        # Subtract the difference to find the last Sunday in September
        last_sunday = last_day_of_september - timedelta(days=days_until_last_sunday)
        # Set the time to 2am
        last_sunday = last_sunday.replace(hour=2, minute=0, second=0, microsecond=0)
        
        return last_sunday

    def get_daylight_saving_ends_date(self, year):
        """
        Daylight saving ends each year at 3am on the first Sunday in April.

        Args:
            year (int): year 'yyyy' that you want to find a NZDT date

        Returns:
            a class 'datetime.datetime'
        """
        # Find the first day of April for the given year
        first_day_of_april = datetime(year, 4, 1)
        # Calculate the difference in days to find the first Sunday
        days_until_first_sunday = (6 - first_day_of_april.weekday()) % 7
        # Add the difference to find the first Sunday in April
        first_sunday = first_day_of_april + timedelta(days=days_until_first_sunday)
        # Set the time to 3am
        first_sunday = first_sunday.replace(hour=3, minute=0, second=0, microsecond=0)
        return first_sunday

    def is_daylight_saving_time(self, dt):
        """
        checking the given date is daylight saving day or not.
        
        Args:
           dt (datetime object): 

        Returns:
            boolean: If date is sitting within the daylight saving period then it will return True, otherwise False.
        """
        # Daylight saving time starts each year at 2am on the last Sunday in September
        start_date = self.get_daylight_saving_starts_date(dt.year)
        # Daylight saving time ends at 3am on the first Sunday in April of the following year
        end_date = self.get_daylight_saving_ends_date(dt.year + 1)
        
        return start_date <= dt < end_date

    def get_adjusted_datetime(self, file_path):
        """
        Within the given .laz data, it fixes the time that has been recorded as GMT into NZDT system depends on regulations
        
        Args:
            file_path (str): the file path of 3d point cloud data which '.laz'

        Returns:
            date (str): yyyy-mm-dd hh:mm:ss.ssssss
        """
        # Read the LAS file
        las_file = laspy.read(file_path)        
        # Extract GPS time
        gps_time = las_file.gps_time
        # Extract the start of GPS time
        start_gps_time = min(gps_time)
        
        # Convert the start_gps_time to a datetime object
        start_datetime = datetime.utcfromtimestamp(start_gps_time)

        # Check if daylight saving time is in effect
        if self.is_daylight_saving_time(start_datetime):
            # Add 13 hours during daylight saving time (12 + 1)
            adjusted_datetime = start_datetime + timedelta(hours=13)
        else:
            # Add 12 hours outside daylight saving time
            adjusted_datetime = start_datetime + timedelta(hours=12)
        # Format the adjusted datetime as a string
        adjusted_formatted_date = adjusted_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')
        
        return adjusted_formatted_date

    def shorted_date(self, datetime_str):
        """
        Original date has milliseconds, so get lid of them to show easily to human
        Args:
            datetime_str (_type_): _description_

        Returns:
            shorter version of date type: dd-mm-yyyy hh:mm:ss
        """
       
        original_datetime = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S.%f')
        formatted_datetime_string = original_datetime.strftime('%d-%m-%Y %H:%M:%S')
        
        return formatted_datetime_string
    
# def main():
#     adjusterObj = DateTimeAdjuster()
#     las_file_path = r'C:\Users\JooHyunAhn\Interpine\AssignedTasks\HovermapUsage\Testing_01\Output\Testing_01_Output_laz1_4.laz'
#     adjusted_date = adjusterObj.get_adjusted_datetime(las_file_path)
#     short = adjusterObj.shorted_date(adjusted_date)
#     print(short)

# if __name__ == "__main__":
#     main()