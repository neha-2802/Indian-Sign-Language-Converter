"""
URL configuration for sign_detection_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path

from django.conf.urls.static import static
from django.conf import settings


from userapp import views as user_views
from adminapp import views as admin_views



urlpatterns = [
    path("admin/", admin.site.urls),
    path("", user_views.index,name="index"),
    path("about/", user_views.about,name="about"),
    path("user-login/", user_views.user_login,name="user_login"),
    path("user-register/", user_views.user_register,name="user_register"),
    path("admin-login/", user_views.admin_login,name="admin_login"),
    path("contact/", user_views.contact,name="contact"),
    path("user-otp/",user_views.user_otp,name="user_otp"),
    path('user-dashboard/',user_views.user_dashboard,name="user_dashboard"),
    path("user/logout/",user_views.user_logout,name="user_logout"),
    path("live-detection/",user_views.live_detection,name="live_detection"),
    path("text-detection/",user_views.text_detection,name="text_detection"),
    path("user-profile/", user_views.user_profile,name="user_profile"),
    path("image-detection/", user_views.image_detection,name="image_detection"),







    path('admin-dashboard/', admin_views.index, name='admin_dashboard'),
    # path('post/remove/<int:post_id>/', admin_views.remove_post, name='remove_post'),
    path('user/change-status/<int:user_id>/', admin_views.change_status, name='change_status'),
    path('all-users/', admin_views.all_users, name='all_users'),
    path('issue-fines/', admin_views.issue_fines, name='issue_fines'),
    path('all-fines/', admin_views.all_fines, name='all_fines'),
    path('upload-dataset/', admin_views.upload_dataset, name='upload_dataset'),
    path('Train-Test-model/', admin_views.trainTestmodel, name='trainTestmodel'),
    path('latest-payments/', admin_views.latest_payments, name='latest_posts'),
    path('cancel-fine/<int:id>/', admin_views.remove_fine, name='remove_fine'),
    path('RM-model/', admin_views.rf, name='rf'),
    path('MD-model/', admin_views.nb, name='mb'),
    path('DM-model/', admin_views.dt, name='dt'),
    path('Graph-analysis/', admin_views.graph, name='graph'),
    path('pending-users/', admin_views.pending_users, name='pending_users'),
    path('accept-user/<int:user_id>/', admin_views.accept_user, name='accept_user'),
    path('reject-user/<int:user_id>/', admin_views.reject_user, name='reject_user'),
    path('delete-user/<int:user_id>/', admin_views.delete_user, name='delete_user'),







]+ static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
