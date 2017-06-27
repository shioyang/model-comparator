import { Component } from '@angular/core'

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

  OnPredictClicked(): void {
    this.result_text += '1'
  }
}
