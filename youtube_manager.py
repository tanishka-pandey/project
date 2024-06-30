import json 

def load_data():
    try :
        with open ('youtube.txt','r') as file :
            t= json.load(file)
            return t
    except FileNotFoundError :
          return []
        
def save_data(videos):
    with open ('youtube.txt','w') as file :
        json.dump(videos,file)
        
def show_all_video(videos):
    print(videos )

        
        
def list_all_video(videos):
    for index,video in enumerate (videos ,start= 1):
        print(f" {index}.{video['name']}, Duration : {video['time']}")
    

def add_video (videos):
    name=input("Enter the video name:")
    time =input("Enter the  video time:")
    videos.append({'name': name, 'time': time})
    save_data(videos)
    
    
def update_video(videos):
    list_all_video(videos)
    index= int(input("Enter the video number u wanted to be updated:"))
    if 1<= index <= len(videos):
        name= input("Enter the new name:")
        time=input("Enter the new time:")
        videos[index-1] = {'name': name , 'time': time}
        save_data (videos)
    else:
        print("You have entered invalid index.")
        
        
    
def delete_video (videos):
    list_all_video(videos)
    index=int(input("Enter the video number u want to be deleted:"))
    if 1<= index <= len (videos):
        del videos[index-1]
        save_data(videos)
    else:
        print("You have entered invalid index.")
        
        

def main():
    videos= load_data()
    while True :
        print("\n Youtube Manager || Choose an option ")
        print("1. Show all video ")
        print("2. List all youtube videos ")
        print("3. Add a youtube video ")
        print("4. Update a youtube video details ")
        print("5. Delete  a youtube video ")
        print("6. Exit the app ")
        choice= input ("Enter your choice: ")
        
        match choice:
            case '1' :
                show_all_video(videos)
            case '2' :
                list_all_video(videos)
            case '3' :
                add_video(videos)
            case '4' :
                update_video(videos)
            case '5' :
                delete_video(videos)
            case '6' :
                break
            case _ :
                print("Oops ! you have entered a wrong choice.")
                
if __name__ == "__main__" :
    
       main()