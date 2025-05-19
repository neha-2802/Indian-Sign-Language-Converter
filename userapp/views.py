from django.shortcuts import render,redirect
from django.contrib import messages
import urllib.request
import urllib.parse
from django.contrib.auth import logout
from django.core.mail import send_mail
import os
import random
from django.conf import settings
from userapp.models import *
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from PIL import Image
from io import BytesIO
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import uuid
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import pyttsx3
import re
import mediapipe as mp

def user_logout(request):
    logout(request)
    messages.info(request, "Logout Successfully ")
    return redirect("user_login")

EMAIL_HOST_USER = os.environ.get('EMAIL_HOST_USER')
EMAIL_HOST_PASSWORD = os.environ.get('EMAIL_HOST_PASSWORD')



def generate_otp(length=4):
    otp = "".join(random.choices("0123456789", k=length))
    return otp

def index(request):
    return render(request,"user/index.html")

def about(request):
    return render(request,"user/about.html")

def user_login(request):
    if request.method == "POST":
        email = request.POST["email"]
        password = request.POST["password"]
        try:
            user = User.objects.get(email=email)
            if user.password != password:
                messages.error(request, "Incorrect password.")
                return redirect("user_login")
            if user.status == "Accepted":
                if user.otp_status == "Verified":
                    request.session["user_id_after_login"] = user.pk
                    messages.success(request, "Login successful!")
                    return redirect("user_dashboard")
                else:
                    new_otp = generate_otp()
                    user.otp = new_otp
                    user.otp_status = "Not Verified"
                    user.save()
                    subject = "New OTP for Verification"
                    message = f"Your new OTP for verification is: {new_otp}"
                    from_email = settings.EMAIL_HOST_USER
                    recipient_list = [user.email]
                    send_mail(
                        subject, message, from_email, recipient_list, fail_silently=False
                    )
                 
                    request.session["id_for_otp_verification_user"] = user.pk
                    return redirect("user_otp")
            else:
                messages.success(request, "Your Account is Not Accepted by Admin Yet")
                return redirect("user_login")
        except User.DoesNotExist:
            messages.error(request, "No User Found.")
            return redirect("user_login")
    return render(request,"user/user-login.html")

def user_register(request):
    if request.method == "POST":
        full_name = request.POST.get('full_name')
        email = request.POST.get('email')
        password = request.POST.get('password') 
        phone_number = request.POST.get('phone_number')
        age = request.POST.get('age')
        address = request.POST.get('address')
        photo = request.FILES.get('photo')
        if User.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return redirect('user_register') 
        user = User(
            full_name=full_name,
            email=email,
            password=password, 
            phone_number=phone_number,
            age=age,
            address=address,
            photo=photo
        )
        otp = generate_otp()
        user.otp = otp
        user.save()
        subject = "OTP Verification for Account Activation"
        message = f"Hello {full_name},\n\nYour OTP for account activation is: {otp}\n\nIf you did not request this OTP, please ignore this email."
        from_email = settings.EMAIL_HOST_USER
        recipient_list = [email]
        request.session["id_for_otp_verification_user"] = user.pk
        send_mail(subject, message, from_email, recipient_list, fail_silently=False)
      
        messages.success(request, "Otp is sent your mail and phonenumber !")
        return redirect("user_otp")
    return render(request,"user/user-register.html")

def admin_login(request):
    if request.method == "POST":
        username = request.POST.get('name')
        password = request.POST.get('password')
        if username == 'admin' and password == 'admin':
            messages.success(request, 'Login Successful')
            return redirect('admin_dashboard')
        else:
            messages.error(request, 'Invalid details !')
            return redirect('admin_login')
    return render(request,"user/admin-login.html")

def contact(request):
    return render(request,"user/contact.html")

def user_otp(request):
    otp_user_id = request.session.get("id_for_otp_verification_user")
    if not otp_user_id:
        messages.error(request, "No OTP session found. Please try again.")
        return redirect("user_register")
    if request.method == "POST":
        entered_otp = "".join(
            [
                request.POST["first"],
                request.POST["second"],
                request.POST["third"],
                request.POST["fourth"],
            ]
        )
        try:
            user = User.objects.get(id=otp_user_id)
        except User.DoesNotExist:
            messages.error(request, "User not found. Please try again.")
            return redirect("user_register")
        if user.otp == entered_otp:
            user.otp_status = "Verified"
            user.save()
            messages.success(request, "OTP verification successful!")
            return redirect("user_login")
        else:
            messages.error(request, "Incorrect OTP. Please try again.")
            return redirect("user_otp")
    return render(request,"user/user-otp.html")

def user_dashboard(request):
    return render(request,"user/user-dashboard.html")

model1 = load_model("sign_model.h5")

class_label = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: 'A', 10: 'B', 11: 'C', 12: 'D',
               13: 'E', 14: 'F', 15: 'G', 16: 'H', 17: 'I', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P',
               25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def edge_detection(image):
    minValue = 70
    blur = cv2.GaussianBlur(image, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return res, contours

def predict_sign(image):
    
    img, contours = edge_detection(image)
    
    if len(contours) == 0:
        return 'invalid'
    
    significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]  
    
    if len(significant_contours) == 0:
        return 'invalid'
    
    img = cv2.resize(img, (64, 64))
    
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    
    predictions = model1.predict(img)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_label[predicted_class]
    
    return predicted_label

def speak_text(text):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

@csrf_exempt
def live_detection(request):
    if request.method == 'POST':
        try:
            
            data = json.loads(request.body)
            image_data = data['image']

            image_data = image_data.split(',')[1]
            image = Image.open(BytesIO(base64.b64decode(image_data)))
            
            image_np = np.array(image.convert('RGB'))  
            
            with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
                results = hands.process(image_np)
                if not results.multi_hand_landmarks:
                    return JsonResponse({'status': 'invalid', 'message': 'No hand detected'}, status=200)
            
            image_np = np.array(image.convert('L'))  

            predicted_label = predict_sign(image_np)
            
            if predicted_label == 'invalid':
                return JsonResponse({'status': 'invalid', 'message': 'No hand sign detected'}, status=200)
            
            speak_text(f'The predicted sign is {predicted_label}')
            
            filename = f"{uuid.uuid4().hex}.png"

            image_io = BytesIO()
            image.save(image_io, format='PNG')
            image_content = ContentFile(image_io.getvalue(), filename)
            file_path = default_storage.save(f'images/{filename}', image_content)

            print(f"Image saved as: {filename}")
            
            return JsonResponse({'status': 'success', 'filename': filename, 'prediction': predicted_label})
        except Exception as e:
            print(f"Error processing image: {e}")
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    return render(request, "user/live-detection.html")

def text_detection(request):
    if request.method == 'POST':
        user_text = request.POST.get('user_text', '')
        filtered_text = re.sub(r'[^A-Za-z1-9]', '', user_text)
        print("Filtered Text:", filtered_text)
        
        base_dir = os.path.join(settings.MEDIA_ROOT, 'sign_images')
        image_paths = []
        
        for char in filtered_text:
            char_folder = char.upper() if char.isalpha() else char
            folder_path = os.path.join(base_dir, char_folder)
            img_path = os.path.join(folder_path, '1.jpg')
            
            print("Checking folder path:", folder_path)
            print("Checking image path:", img_path)
            
            if os.path.isfile(img_path):
                img_url = f'{settings.MEDIA_URL}sign_images/{char_folder}/1.jpg'
                print("Image URL:", img_url) 
                image_paths.append(img_url)
            else:
                print("File not found:", img_path) 
        
        return JsonResponse({'images': image_paths})
        
    return render(request, "user/text-detection.html")

from django.utils.datastructures import MultiValueDictKeyError

def user_profile(request):
    user_id  = request.session.get('user_id_after_login')
    print(user_id)
    user = User.objects.get(pk= user_id)
    if request.method == "POST":
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        try:
            profile = request.FILES['profile']
            user.photo = profile
        except MultiValueDictKeyError:
            profile = user.photo
        password = request.POST.get('password')
        location = request.POST.get('location')
        user.user_name = name
        user.email = email
        user.phone_number = phone
        user.password = password
        user.address = location
        user.save()
        messages.success(request , 'updated succesfully!')
        return redirect('user_profile')
    return render(request,'user/user-profile.html',{'user':user})



@csrf_exempt
def image_detection(request):
    if request.method == 'POST':
        images = request.FILES.getlist('images')
        results = []
        predicted_labels = "" 
        
        for image_file in images:
            # Read image file
            img = Image.open(image_file)
            img = img.convert('L')
            img = np.array(img)
            
            # Predict sign
            prediction = predict_sign(img)
            
            predicted_labels += prediction

            # Convert image to base64 string
            buffered = BytesIO()
            img_pil = Image.fromarray(img)
            img_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Save the prediction and image data
            result = {
                'prediction': prediction,
                'image': img_str,
            }
            results.append(result)
        
        return JsonResponse({'results': results, 'predicted_labels': predicted_labels}, safe=False)
    
    return render(request, "user/image-detection.html")