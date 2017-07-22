import { Component } from '@angular/core'
import { ServerService } from './server.service'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'app'
  images = [
    {
      name: 'Pred #1',
      file_name: '1_pred.jpg',
      path: './ml/data_prediction/1_pred.jpg'
    },
    {
      name: 'Pred #2',
      file_name: '2_pred.jpg',
      path: './ml/data_prediction/2_pred.jpg'
    },
    {
      name: 'Pred #3',
      file_name: '3_pred.jpg',
      path: './ml/data_prediction/3_pred.jpg'
    }
  ]
  selected_image = ''
  result_text = ''

  constructor(private serverService: ServerService){}

  OnPredictClicked(): void {
    let image_path = this.selected_image.substring(5);
    console.log('image_path: ' + image_path)
    this.serverService.predictImage(image_path)
      .subscribe(result => {
        this.result_text = result
      })
  }
}
