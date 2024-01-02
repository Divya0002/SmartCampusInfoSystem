# Student Information System for Android

This Android application provides a Student Information System with features for student login and a home screen displaying various functionalities.

## Students Login (StudentsActivity)

The `StudentsActivity` class handles the login functionality for students. It includes the following components:

- Username and password input fields
- HTTP request to a server for authentication
- Generation and sending of OTP (One Time Password) via SMS
- Redirects to OTP verification screen upon successful login

## Students Home (StudentsHomeActivity)

The `StudentsHomeActivity` class represents the home screen for students after successful login. It includes buttons for quick access to different sections:

1. Academic Information
2. Results
3. Attendance
4. Placement Details
5. Extra-Curricular Activities
6. Logout

## Usage

Clone the repository and import the Android project into Android Studio. Ensure the necessary dependencies are installed and configure the server URL accordingly. Run the application on an Android emulator or a physical device.

## Dependencies

- Android SDK
- Android Studio
- Java

## Contributing

Feel free to contribute by submitting bug reports, feature requests, or pull requests. Please follow the code of conduct and contribution guidelines.

## License

This project is licensed under the [MIT License](LICENSE).
