#function for applying median filter
def medianFilter(imagePath):
  img = cv2.imread(imagePath)
  width, height= img.size

  #create an array for resultant image
  result = np.zeros((height, width))

  #traverse through the pixels of the image
  for i in range(width):
    for j in range(height):

      #initialize variables
      currentElement=0; left=0; right=0; top=0; bottom=0; topLeft=0; 
      topRight=0; bottomLeft=0; bottomRight=0;          

      #get current pixel
      currentElement = img.getpixel((i,j))
      
      #offset is equal to 1 in a 3x3 filter
      offset=1
      
      #get left, right, bottom and top pixels
      #with respect to current pixel
      if not i-offset < 0:
        left = img.getpixel((i-offset,j))                        
      if not i+offset > width-offset:
        right = img.getpixel((i+offset,j))
      if not j-offset < 0:
        top = img.getpixel((i,j-offset))
      if not j+offset > height-1:
        bottom = img.getpixel((i,j+offset))

      #get top left, top right, bottom left and bottom right pixels
      #with respect to current pixel
      if not i-offset < 0 and not j-offset < 0:
        topLeft = img.getpixel((i-offset,j-offset))
      if not j-offset < 0 and not i+offset > width-1:
        topRight = img.getpixel((i+offset,j-offset))
      if not j+offset > height-1 and not i-offset < 0:
        bottomLeft = img.getpixel((i-offset,j+offset))
      if not j+offset > height-1 and not i+offset > width-1:
        bottomRight = img.getpixel((i+offset,j+offset))
      
      #get median of all pixels retrieved 
      med=statistics.median([currentElement,left,right,top,bottom,topLeft,topRight,bottomLeft,bottomRight])
      
      #put median in the same position in resultant array
      result[j][i] = med
    
  #return resultant array  
  return result

