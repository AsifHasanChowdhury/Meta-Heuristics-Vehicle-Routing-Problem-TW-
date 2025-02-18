<strong> Welcome to the Git Repo of E-Vehicle Routing Problem </strong>

- Step 1: Clone the git repository
- Step 2: Locate the GAElitism.py file. This is the driver file.
- Step 3: Install the necessary packages using Pip Commands.
- Step 4: Locate the dataset folder and setup the dataset path Accordingly (Local Machine Base Directory Setup) 
  - Please note that you have to change route details like C101, R101 manually in the code.
- Step 5: Open Terminal from VS Code and type GAElitism.py or Click the Play button At the bottom Corner.


- <strong> Codebase Walkthrough <strong>
- Step 1: Clone the git repository & you will find the code base like the image below.
   <img width="960" alt="{18BAC09F-1C4F-46E5-8935-72DEDEB1E6AB}" src="https://github.com/user-attachments/assets/a54afdca-6d2c-4b88-852d-09f0dbfb1840" />

- Step 2: Make sure All the necessary libraries Are installed.
  - You can install the necessary package using CMD/ git bash, as you choose. We will show the installation guidelines using the terminal from Visual Studio code.
  - from visual studio code terminal type !pip install torch
  - from visual studio code terminal type !pip install numpy
  - from visual studio code terminal type !pip install matplotlib
  - from visual studio code terminal type !pip install jsonlib
  - IF ALL DEPENDENCIES ARE RESOLVED THEN YOU WILL SEE THE LIBRARY LINES WITHOUT ANY ERROR OR WARNING LIKE THE IMAGE BELOW
 
    <img width="378" alt="{B88F9E44-FE05-4A90-BB94-97DDAE3DBE99}" src="https://github.com/user-attachments/assets/e3f958f6-326c-41b2-8fc3-8c7977ab2f91" />

  -Step 3: Check the FileSystem.py as the image below. You will find a method called filepath. Setup the folder location of your dataset Accordingly.

    <img width="958" alt="{888E4511-1024-46B6-B23B-57D7C40A9A14}" src="https://github.com/user-attachments/assets/f1a5b679-5b7a-434e-b35a-2d79e145418c" />

  -Step 4: Click the Play button At the top-most corner to run the codebase. You can Also type GAElitism.py in the terminal to run it. If it runs properly then you will see the console like the image below.

    <img width="793" alt="{14846BB0-2B8C-4FDD-9850-6598226807BE}" src="https://github.com/user-attachments/assets/6400be2f-a39b-4299-924b-241f3c3b3c03" />

  - Step 5: In order to monitor the result make sure you change the instance name properly and keep it similar in both places as per the pictures below.
    -- First Place is in GAElistm.py file.
    -- Second Place is in FileSystem.py file.
    <img width="959" alt="{EC786E9B-2E62-4E6D-AC97-0BD824EB4427}" src="https://github.com/user-attachments/assets/20c53f8a-462d-4aec-9086-6249b35993ce" />
    <img width="897" alt="{D10D0C98-E8B4-4A07-A03F-43DFF86D0D16}" src="https://github.com/user-attachments/assets/24de8e16-7ba2-45bf-98fd-be6ee1cfdefa" />
    -- Once the code ran properly. Locate the result folder in your local machine for CSV files like the picture below. A new file will be created everytime if you change the instance properly.
    <img width="554" alt="{398FB12D-2162-4B14-B376-B3C0E0F44394}" src="https://github.com/user-attachments/assets/f3141c79-6983-45c8-9ec2-68267a244b93" />


