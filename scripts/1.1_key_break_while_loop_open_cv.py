# Not working in pycharm

import cv2
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Key 'Q' is pressed. Breaking loop...")
        break

print("We are outside while loop")

    # while True:
#     print("While loop")
#     # press 'q' to exit
#     k = cv2.waitKey(0)
#     if k == 27:  # wait for ESC key to exit and terminate program,
#         cv2.destroyAllWindows()
#         print("You pressed a key!")
#         break  # exit loop
#         print("After break...")
#
#
#
# cv2.destroyAllWindows()
# print("We are outside while loop")

