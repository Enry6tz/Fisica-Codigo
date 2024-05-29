import cv2

class Tracker:
    def __init__(self):
        self.lX = []
        self.lY = []
        self.lP = []
        self.lT = []
        self.deltaT = 0
        self.ALTURA =0
        self.ANCHO=0

    def calcularVelocidades (self,p0,p1,p2):
        vx = (p0[0] - p1[0])/self.deltaT
        vy = (p0[1] - p1[1])/self.deltaT

        vx2 = (p1[0] - p2[0])/self.deltaT
        vy2 = (p1[1] - p2[1])/self.deltaT

        return vx,vy,vx2,vy2

    def calcularAceleraciones (self,vx,vy,vx2,vy2):
        ax = (vx-vx2)/self.deltaT
        ay = (vy-vy2)/self.deltaT

        return ax,ay
    
    def track(self,videoPath,deltaT,show):
        self.videoPath= videoPath
        tick=tIni=tFin=0
        match(videoPath):
            case "Tiro_encesto1.mp4":
                tIni = 2
                tFin = 37
                bbox = (304,493,44,48)
            case "Tiro_encesto2.mp4":
                tIni = 1
                tFin = 37
                bbox = (483,476,35,36)
            case "Tiro_encesto3.mp4":
                tIni = 8
                tFin = 47
                bbox = (345, 399, 25, 25)       
            case "Tiro_erro1.mp4":
                tIni = 1
                tFin = 40
                bbox = (496,528,35,39)
        
        
        # #obtenemos un deltaT fijo que depende de la longitud del video y de su framerate
        # #frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_count = tickFin-tickInicio
        # duration = frame_count/fps
        # # deltaT = duration/fps
        
        self.tracker = cv2.TrackerCSRT_create()
        video = cv2.VideoCapture("Practico\\Videos\\"+videoPath)
        ok, frame = video.read()
        ok = self.tracker.init(frame, bbox)

        self.ALTURA, self.ANCHO, _ = frame.shape
        fps = video.get(cv2.CAP_PROP_FPS)
        self.deltaT=1/fps
        out = cv2.VideoWriter('Practico\\OutputVideos\\'+videoPath, cv2.VideoWriter_fourcc(*'XVID'), fps, (self.ANCHO, self.ALTURA),False)

        p2=p1=p0=(0,0)

        tiempoActual=0
        
        while(True):
            tick+=1    
            ok, frame = video.read()
            if(ok and tIni<=tick<tFin):
                ok, bbox = self.tracker.update(frame)
                centre1 = (int(bbox[0]), int(bbox[1]))
                centre2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, centre1, centre2, (255,0,0), 2, 1)
                
                p2=p1
                p1=p0
                p0=(int(bbox[0]+(bbox[2]/2)),int(bbox[1]+(bbox[3]/2)))
                
                tiempoActual += self.deltaT
                self.lT.append(tiempoActual.__round__(3))
                self.lP.append(p0)
                self.lX.append(p0[0])
                self.lY.append(p0[1])

                if(p2!=(0,0)):
                    vx,vy,vx2,vy2 = self.calcularVelocidades(p0,p1,p2)
                    ax,ay = self.calcularAceleraciones(vx,vy,vx2,vy2)
                    cv2.arrowedLine(frame,p0,(int(p0[0]+vx/2),int(p0[1]+vy/2)),(160,0,160),3)
                    cv2.arrowedLine(frame,p0,(int(p0[0]),int(p0[1]+ay/2)),(230,230,0),3)
                
                out.write(frame)
                if(show):
                    cv2.imshow("GinobiLib", frame)
                    cv2.setWindowProperty("GinobiLib", cv2.WND_PROP_TOPMOST, 1)

                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 :
                    break
            
            if(not ok):
                break
        
        out.release()
        video.release()
        return out

    def getSize(self):
        return self.ALTURA,self.ANCHO
    
    def getLists(self):
        return self.lX,self.lY,self.lP,self.lT
